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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from utils.constants import IGNORE_INDEX



# LLaGA 配置类，继承自 LlamaConfig，用于注册新的模型类型 "llaga"
class LlagaConfig(LlamaConfig):
    model_type = "llaga"


# LLaGA 模型基类
# 继承自 LlagaMetaModel (处理图模态的元类) 和 LlamaModel (HuggingFace 的 Llama 基础模型)
# 这种多重继承使得该模型既拥有处理 Graph Projector 的能力，又拥有 Llama 的 Transformer 结构
class LlagaLlamaModel(LlagaMetaModel, LlamaModel):
    config_class = LlagaConfig

    def __init__(self, config: LlamaConfig):
        super(LlagaLlamaModel, self).__init__(config)


# LLaGA 因果语言模型 (用于生成/对话任务)
# 继承自 LlamaForCausalLM (带 LM Head 的 Llama) 和 LlagaMetaForCausalLM (处理图模态输入/输出的元类)
class LlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig


    def __init__(self, config):
        # 初始化父类 LlamaForCausalLM
        #先继承，然后替换掉主模型
        #这里调用的哪个初始化的方法
        super(LlamaForCausalLM, self).__init__(config)
        # 替换内部的主干模型为支持图模态的 LlagaLlamaModel

        self.model = LlagaLlamaModel(config)

        # 定义语言建模头 (LM Head)，将隐层状态映射回词表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        # 返回内部的主干模型，LlagaMetaForCausalLM 中会用到它来获取 projector
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[list] = None, # 接收图结构/ID索引
        graph_emb: Optional[list] = None, # 接收图的原始 Embedding
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # [关键步骤] 准备多模态输入和标签
        # 该方法来自 LlagaMetaForCausalLM。
        # 它会：
        # 1. 将 input_ids 中的 <graph> token 替换为经过 Projector 映射后的图特征 Embedding。
        # 2. 相应地调整 attention_mask 和 labels，以匹配新的序列长度（因为一个 <graph> 可能对应多个 embedding 向量）。
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        
        
        # 将处理好的 inputs_embeds (包含了文本和图特征) 输入给 LLaMA 主干模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # 计算下一个 token 的预测概率分布
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 标准的自回归语言模型 Loss 计算：
            # 预测第 t 个 token 的 logits 应该对应第 t+1 个 token 的 label
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 准备生成阶段的输入
    # 在生成 response 时，每一步都会调用此函数来准备当前的输入 token
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            # 如果有 KV Cache，只需要输入最后一个 token
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果传入了 embedding（即第一步，包含图特征的 embedding），直接使用它
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 确保图相关参数也能传递（虽然主要在第一步 prepare_inputs_labels_for_multimodal 中使用）
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )
        return model_inputs

AutoConfig.register("llaga", LlagaConfig)
AutoModelForCausalLM.register(LlagaConfig, LlagaLlamaForCausalLM)
