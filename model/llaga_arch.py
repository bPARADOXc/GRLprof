#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn
import re

# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_vision_projector

from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID


def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class GraphPathEncoder(nn.Module):
    """
    将知识图谱推理路径编码为 LLM token embedding。

    输入（每个样本）:
        sample_paths: List[List[Tensor(3, graph_emb_dim)]]
            外层列表: N 条路径
            内层列表: 每条路径最多 MAX_TRIPLES 个三元组
            每个 Tensor 行含义: [头实体, 关系, 尾实体]

    处理流程（每条路径）:
        1. 位置向量: 对每个三元组计算 h+r-t, pad 至 MAX_TRIPLES 个向量后
           展平拼接, 经 Linear(MAX_TRIPLES*d, d) 得到位置向量 (d,)
        2. 文本向量:
           - 分别收集头实体/关系/尾实体向量, pad 并展平
           - 各自经 2 层 MLP (MAX_TRIPLES*d -> d) 得到 3 个 (d,) 向量
           - 拼接后经 Linear(3d, d) 得到文本向量 (d,)
        3. 路径信息 token:
           - 将 [位置向量, 文本向量] 构成长度为 2 的序列
           - 经小型 Transformer Encoder (2 层, d_model=d) 后均值池化得 (d,)
        4. 输出投影: 经 2 层 MLP (d -> output_dim) 得到最终 token (output_dim,)

    输出: Tensor(N, output_dim)
    """

    MAX_TRIPLES = 4

    def __init__(self, graph_emb_dim: int = 768, output_dim: int = 4096):
        super().__init__()
        d = graph_emb_dim
        D = self.MAX_TRIPLES * d  # 展平后的拼接维度 (3072)

        # --- 位置分支 ---
        self.position_fc = nn.Linear(D, d)

        # --- 文本分支 (头实体 / 关系 / 尾实体) ---

#         用 d 作为中间维度的好处：

# 参数量和显存开销明显更小，训练更稳。
# 你的输入含大量 pad，D 空间冗余高，继续在 D 上做隐藏层收益通常不大。
# 整体架构目标是把路径信息映射到统一 768 语义空间，保持 d 作为隐层更一致。
# 什么时候考虑 D：

# 只有在你确认模型欠拟合、数据量足够大、算力充足时，才考虑更宽隐层。
# 实践上更推荐先试 D 到 2d 到 d，而不是直接用 D 到 D 到 d。
        self.head_mlp = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
        self.relation_mlp = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
        self.tail_mlp = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))

        # 三段文本向量合并
        self.text_linear = nn.Linear(3 * d, d)

        # --- 小型 Transformer Encoder (序列长度=2) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        #层数8
        self.path_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # --- 输出 MLP ---
        self.output_mlp = nn.Sequential(
            nn.Linear(d, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        self.graph_emb_dim = graph_emb_dim
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    def _pad_concat(self, vecs: list, device, dtype) -> torch.Tensor:
        """将至多 MAX_TRIPLES 个 (d,) 向量补零后展平, 返回 (MAX_TRIPLES*d,)."""
        #device要保持一致，不然cat不了
        d = self.graph_emb_dim
        pad_count = self.MAX_TRIPLES - len(vecs)
        if pad_count > 0:
            zeros = torch.zeros(pad_count, d, device=device, dtype=dtype)
            vecs = list(vecs) + [zeros[i] for i in range(pad_count)]#列表里面是tensor，注意zeros取出来的是一维张量
        return torch.cat(vecs[: self.MAX_TRIPLES], dim=0)

    def encode_path(self, triples: list) -> torch.Tensor:
        """
        triples: List[Tensor(3, graph_emb_dim)], 至多 MAX_TRIPLES 个三元组
        返回: (output_dim,)
        """
        device = triples[0].device
        dtype = self.position_fc.weight.dtype  # 跟随模型精度

        pos_vecs, head_vecs, rel_vecs, tail_vecs = [], [], [], []
        for triple in triples:
            triple = triple.to(device=device, dtype=dtype)
            h, r, t = triple[0], triple[1], triple[2]
            pos_vecs.append(h + r - t)
            head_vecs.append(h)
            rel_vecs.append(r)
            tail_vecs.append(t)

        # 位置向量
        position_vec = self.position_fc(
            self._pad_concat(pos_vecs, device, dtype)
        )  # (d,)

        # 文本向量
        head_text = self.head_mlp(self._pad_concat(head_vecs, device, dtype))
        rel_text = self.relation_mlp(self._pad_concat(rel_vecs, device, dtype))
        tail_text = self.tail_mlp(self._pad_concat(tail_vecs, device, dtype))
        text_vec = self.text_linear(
            torch.cat([head_text, rel_text, tail_text], dim=0)
        )  # (d,)

        # 小型 Encoder: [位置向量, 文本向量] → (1, 2, d)
        seq = torch.stack([position_vec, text_vec], dim=0).unsqueeze(0)
        encoded = self.path_encoder(seq)  # (1, 2, d)
        path_token = encoded[0].mean(dim=0)  # (d,) — 均值池化

        return self.output_mlp(path_token)  # (output_dim,)

    def forward(self, sample_paths: list) -> torch.Tensor:
        """
        sample_paths: List[List[Tensor(3, graph_emb_dim)]]  — 一个样本的所有路径
        返回: Tensor(N, output_dim), N 为路径条数
        """
        max_graph_paths = max(0, int(os.environ.get("LLAGA_MAX_GRAPH_PATHS", "16")))
        if max_graph_paths > 0 and len(sample_paths) > max_graph_paths:
            sample_paths = sample_paths[:max_graph_paths]

        if len(sample_paths) == 0:
            return torch.zeros(
                (0, self.output_dim),
                device=self.position_fc.weight.device,
                dtype=self.position_fc.weight.dtype,
            )

        # 路径条数可能非常大，逐路径调用 encode_path 会触发大量 Python 循环和小张量算子，
        # 在 WebQSP 这种长路径样本上会非常慢。这里把一个样本内的所有路径打包成批量张量，
        # 保持计算逻辑不变，但让 Linear / Transformer 一次性处理整批路径。
        device = self.position_fc.weight.device
        dtype = self.position_fc.weight.dtype
        max_triples = self.MAX_TRIPLES
        d = self.graph_emb_dim
        path_batch_size = max(1, int(os.environ.get("LLAGA_GRAPH_PATH_BATCH", "16")))

        def encode_chunk(paths_chunk: list) -> torch.Tensor:
            num_paths = len(paths_chunk)
            heads = torch.zeros((num_paths, max_triples, d), device=device, dtype=dtype)
            relations = torch.zeros((num_paths, max_triples, d), device=device, dtype=dtype)
            tails = torch.zeros((num_paths, max_triples, d), device=device, dtype=dtype)

            for path_idx, path in enumerate(paths_chunk):
                if len(path) == 0:
                    continue
                triple_tensors = [
                    triple.to(device=device, dtype=dtype)
                    for triple in path[:max_triples]
                ]
                stacked = torch.stack(triple_tensors, dim=0)
                triple_count = stacked.shape[0]
                heads[path_idx, :triple_count] = stacked[:, 0]
                relations[path_idx, :triple_count] = stacked[:, 1]
                tails[path_idx, :triple_count] = stacked[:, 2]

            position_inputs = (heads + relations - tails).reshape(num_paths, max_triples * d)
            position_vec = self.position_fc(position_inputs)

            head_text = self.head_mlp(heads.reshape(num_paths, max_triples * d))
            rel_text = self.relation_mlp(relations.reshape(num_paths, max_triples * d))
            tail_text = self.tail_mlp(tails.reshape(num_paths, max_triples * d))
            text_vec = self.text_linear(torch.cat([head_text, rel_text, tail_text], dim=-1))

            seq = torch.stack([position_vec, text_vec], dim=1)
            # PyTorch 在 eval/no_grad 下会让 TransformerEncoderLayer 走 fused MHA fastpath。
            # ZeRO-3 分片参数在这条路径上可能表现为 1-D shard，触发
            # "expected 2-D qkv_weight, got 1-D tensor"。这里局部关闭 fastpath，
            # 保留批量化收益，同时避免验证阶段的 fused-kernel 兼容性问题。
            restore_eval_mode = not self.path_encoder.training
            if restore_eval_mode:
                # 某些旧版 PyTorch 没有 torch.backends.mha 开关，但只要 encoder 处于 eval/no_grad，
                # 仍可能走到 _transformer_encoder_layer_fwd 的 fused fastpath。
                # 这里临时切到 train 模式来稳定绕开该路径；本模块 dropout=0.0，所以不会引入随机性。
                self.path_encoder.train()
            mha_backend = getattr(torch.backends, "mha", None)
            mha_fastpath_enabled = (
                mha_backend is not None
                and hasattr(mha_backend, "get_fastpath_enabled")
                and hasattr(mha_backend, "set_fastpath_enabled")
                and mha_backend.get_fastpath_enabled()
            )
            if mha_fastpath_enabled:
                mha_backend.set_fastpath_enabled(False)
            try:
                encoded = self.path_encoder(seq)
            finally:
                if mha_fastpath_enabled:
                    mha_backend.set_fastpath_enabled(True)
                if restore_eval_mode:
                    self.path_encoder.eval()
            path_token = encoded.mean(dim=1)
            return self.output_mlp(path_token)

        if len(sample_paths) <= path_batch_size:
            return encode_chunk(sample_paths)

        outputs = []
        for start in range(0, len(sample_paths), path_batch_size):
            outputs.append(encode_chunk(sample_paths[start:start + path_batch_size]))
        return torch.cat(outputs, dim=0)


class LlagaMetaModel:
    def __init__(self, config):
        super(LlagaMetaModel, self).__init__(config)

        if hasattr(config, "mm_hidden_size"):
            output_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 4096))
            self.graph_path_encoder = GraphPathEncoder(
                graph_emb_dim=config.mm_hidden_size,
                output_dim=output_dim,
            )
            #这个不会触发，看是否需要加
        if getattr(config, "mm_use_graph_start_end", False):
            self.initialize_graph_start_end_embeddings()
        if hasattr(config, "mm_use_graph_special_token") and getattr(config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()


    def initialize_graph_modules(self, model_args, fsdp=None):
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'mlp')
        self.config.mm_hidden_size = getattr(model_args, 'mm_hidden_size')

        output_dim = getattr(self.config, 'word_embed_proj_dim', getattr(self.config, 'hidden_size', 4096))
        self.graph_path_encoder = GraphPathEncoder(
            graph_emb_dim=self.config.mm_hidden_size,
            output_dim=output_dim,
        )
        #这里加入图token
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()


    def initialize_graph_start_end_embeddings(self, init_weight=None):
        input_embeddings = self.get_input_embeddings().weight
        if init_weight is None:
            init_weight = input_embeddings.data.mean(dim=0, keepdim=True).repeat(2, 1)
        else:
            init_weight = init_weight.detach().clone()
            if init_weight.shape[0] != 2:
                init_weight = init_weight[-2:]

        init_weight = init_weight.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        self.graph_start_end_emb = nn.Parameter(init_weight, requires_grad=True)
        self.config.use_separate_graph_start_end_emb = True


##要不要用这个
    def build_special_tokens(self):#初始化的时候创建
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            num_token=self.config.use_hop+2
            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).unsqueeze(1).detach()
            #中间为什么要保留一个维度
            special_token_emb=torch.nn.parameter.Parameter(data=input_embeddings_avg.repeat(num_token, 1, 1), requires_grad=True)
            return special_token_emb
        return None

class LlagaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def encode_graphs(self, graphs, graph_emb):
        """
        graphs    : List[str]  — 每个样本的图描述字符串（元数据，不参与编码）
        graph_emb : List[List[List[Tensor(3, d)]]]  — batch × 路径 × 三元组
        返回: List[Tensor(N_i, output_dim)]，N_i 为第 i 个样本的路径条数
        """
        #所以这里的get——model是子类的方法，继承关系
        encoder = self.get_model().graph_path_encoder
        return [encoder(sample_paths) for sample_paths in graph_emb]

    def inject_special_token(self, graph_emb):#推理时插入这些特殊token
        use_hop=self.config.use_hop
        sample_size = self.config.sample_neighbor_size
        assert graph_emb.shape[-2] == int((sample_size ** (use_hop + 1) - 1) / (sample_size - 1))
        assert self.model.special_token_emb.shape[0] == use_hop + 2
        new_graph_emb = []
        new_graph_emb.append(self.model.special_token_emb[0])
        cur=0
        for i in range(use_hop+1):
            cur_size = sample_size**i
            new_graph_emb.append(graph_emb[cur:cur+cur_size])
            cur+=cur_size
            new_graph_emb.append(self.model.special_token_emb[i+1])
        new_graph_emb = torch.concat(new_graph_emb, dim=0)
        return new_graph_emb

    def _graph_start_end_embeds(self, cur_input_ids, graph_token_start, reference_tensor):
        graph_start_end_emb = getattr(self.get_model(), "graph_start_end_emb", None)
        if graph_start_end_emb is None:
            start_embed = self.get_model().embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start])
            end_embed = self.get_model().embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2])
            return start_embed, end_embed

        graph_start_end_emb = graph_start_end_emb.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        return graph_start_end_emb[0:1], graph_start_end_emb[1:2]

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb
    ):
        #历史 cache 长度是 past_key_values[-1][-1].shape[-2]
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        graph_features = self.encode_graphs(graphs, graph_emb)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            #逐元素判断，并返回同形状的张量
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:#循环处理图token，确保一个样本中只有一个图token
                cur_graph_features = graph_features[cur_graph_idx]
                #这个要不要加上
                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)
                #图start/end token需要训练，但图token需要吗，不需要，没走emb
                graph_token_start = graph_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    graph_start_embed, graph_end_embed = self._graph_start_end_embeds(
                        cur_input_ids, graph_token_start, cur_graph_features
                    )
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())#除了新增的两个，其他都没训练，被detach了
                    cur_new_input_embeds.append(graph_start_embed)
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(graph_end_embed)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # The embedding after graph_features is </GH>, so align it
                        # with the end-token label rather than the <graph> placeholder.
                        cur_new_labels.append(cur_labels[graph_token_start+1:graph_token_start+2])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)#列表append有问题
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[graph_token_start+1:]
                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
#把 batch 里每个样本的 new_input_embeds、new_labels、attention_mask 对齐成统一长度，方便后续 stack 成标准 batch 张量。
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                #左边补true，右边补false
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    # 另一个版本的输入准备函数，支持更复杂的 Attention Mask 处理
    # 区别于上一个函数：它分别计算每个样本的 embedding 和 mask，最后再进行 Padding 和 Stack。
    # 这种方式对于处理变长序列和复杂的 Mask 逻辑可能更稳健。
    #认真计算attentionmask，而不是通过减法计算
    def prepare_inputs_labels_for_multimodal_with_pad_mask(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb
    ):
        #如果有历史缓存且当前输入只有一个token（通常是推理阶段的增量生成），直接扩展attention mask，返回原始输入，不做多模态处理。
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:#只输入一个token
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        graph_features = self.encode_graphs(graphs, graph_emb)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_attention_masks = []
        cur_graph_idx = 0
        
        # 逐样本处理
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_attention_mask = attention_mask[batch_idx]
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                # 同上，处理无图样本的深层兼容性 hack
                half_len = cur_input_ids.shape[0] // 2#向下取整
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_attn_masks=[]
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
                
            # 处理图 Token 替换
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                # graph 现在是字符串，所有路径 token 均有效，掩码全为 True
                cur_graph_mask = torch.ones(
                    cur_graph_features.shape[0], dtype=torch.bool,
                    device=cur_graph_features.device
                )
                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)

                graph_token_start = graph_token_indices[0]
                
                # 拼接逻辑：文本 -> 图 -> 文本
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    # Adapter 微调模式 + Start/End Token
                    graph_start_embed, graph_end_embed = self._graph_start_end_embeds(
                        cur_input_ids, graph_token_start, cur_graph_features
                    )
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    cur_new_input_embeds.append(graph_start_embed)
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(graph_end_embed)
                    
                    # 同时切分Attention Mask
                    cur_attn_masks.append(cur_attention_mask[:graph_token_start])
                    cur_attn_masks.append(cur_graph_mask)
                    cur_attn_masks.append(cur_attention_mask[graph_token_start+1:graph_token_start+2])
                    
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # The embedding after graph_features is </GH>, so align it
                        # with the end-token label rather than the <graph> placeholder.
                        cur_new_labels.append(cur_labels[graph_token_start+1:graph_token_start+2])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:
                    # 普通模式
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_attn_masks.append(cur_attention_mask[:graph_token_start])
                    cur_attn_masks.append(cur_graph_mask)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[graph_token_start+1:]

                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                    cur_attention_mask = cur_attention_mask[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                    cur_attention_mask = cur_attention_mask[graph_token_start + 1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            
            # 添加剩余部分
            if cur_input_ids.numel() > 0:

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                cur_attn_masks.append(cur_attention_mask)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_attn_masks = [x.to(device=self.device) for x in cur_attn_masks]
            cur_attn_masks = torch.cat(cur_attn_masks, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_attention_masks.append(cur_attn_masks)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(new_attention_masks, _new_labels, new_labels):
                    assert cur_attention_mask.shape == cur_new_labels.shape
                    # new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            attention_mask = torch.stack(new_attention_masks, dim=0)
            assert attention_mask.shape == new_input_embeds.shape[:2]
            # if attention_mask is not None:
            #     new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            #     attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            #     assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

# 为分词器和模型添加 <GH>、</GH> 等图结构特殊 token。
# 合理初始化新 token 的嵌入参数（均值或预训练权重）。
# 根据训练模式设置参数的可训练性。
    # 初始化图相关的 Tokenizer 设置
    # 作用：
    # 1. 扩展词表：添加特殊的 <graph_start>, <graph_end> token。
    # 2. 智能初始化：利用现有词向量的均值来初始化新 token，避免随机初始化的参数导致训练初期 loss 剧烈波动。
    # 3. 调整参数冻结策略：只训练独立的 graph_start_end_emb，避免把整张 input embedding 表交给优化器。
    def initialize_graph_tokenizer(self, model_args, tokenizer):
#向词表末尾追加两个新词（通常是 <graph_start> 和 <graph_end>）。
        if model_args.mm_use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
# 目的：避免随机初始化。新加的 token 默认是随机初始化的，这在训练初期会导致模型极不稳定（Loss 爆炸）。
# 做法：计算原有词表中所有词向量的均值 (Mean)，然后用这个均值来初始化新加入的两个 Token。这是一种常见的“软着陆”策略。
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            graph_token_ids = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN])
            if all(token_id is not None and token_id >= 0 for token_id in graph_token_ids):
                graph_token_init = self.get_input_embeddings().weight.data[graph_token_ids]
            else:
                graph_token_init = self.get_input_embeddings().weight.data[-2:]
            self.get_model().initialize_graph_start_end_embeddings(graph_token_init)

# 背景：tune_mm_mlp_adapter 模式通常意味着我们冻结了大模型主干，只训练 Projector。
# 这里保持整张 Input Embedding 冻结，只训练独立的 graph_start_end_emb 两行参数。
# 这样仍能学习 <GH> / </GH> 的输入向量，但不会让 ZeRO-3 把完整词表 embedding 当作可训练参数。
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                self.get_model().graph_start_end_emb.requires_grad = True

            # 如果有预训练的 adapter 权重，尝试加载对应的 Token Embedding
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                graph_token_weight = mm_projector_weights.get('model.graph_start_end_emb')
                if graph_token_weight is None:
                    for legacy_key in (
                        'model.embed_tokens.weight',
                        'model.model.embed_tokens.weight',
                        'base_model.model.model.embed_tokens.weight',
                    ):
                        if legacy_key in mm_projector_weights:
                            graph_token_weight = mm_projector_weights[legacy_key][-2:]
                            break
                if graph_token_weight is not None:
                    self.get_model().initialize_graph_start_end_embeddings(graph_token_weight)
