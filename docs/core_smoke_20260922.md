# 原系统短链路验证（2026-09-22）

## 范围与配置

本次只检查代码可运行、训练可反向传播及保存加载，不做完整训练，
不据此判断收敛或论文指标。GPU 测试仅使用物理 GPU2。

按用户确认保留原流程：准备好的问题、主题实体及候选图 →
三元组检索（GTE-large-en-v1.5，1024 维）→ 路径恢复 →
GTE-ModernBERT（768 维）→ 结构感知路径映射器 → 冻结的 Llama3-8B-Instruct。
不运行实体链接、在线 Freebase 子图获取；对应代码保留。
没有改动路径编码器的八层结构、原 checkpoint 或原始数据。

## 数据边界

- 第一阶段输入是 `ml1996/webqsp` / `rmanluo/RoG-cwq` 的原始记录：
  `id, question, q_entity, a_entity, answer, graph`；`graph` 为候选三元组列表。
- `webqsp_may_merged` 和 `llaga_cwq_may_merged` 是第二阶段输入，
  含 `reasoning_paths`，配套按样本 ID 存储的 768 维 ModernBERT 路径缓存。
- WebQSP 第二阶段使用本地准备好的 `webqsp_may_merged`：
  train/validation/test 为 2717/235/1639 条。
- CWQ 第二阶段使用
  本地准备好的 `llaga_cwq_may_merged`：
  22089/2848/3531 条。没有使用本地另一份训练路径为空的 CWQ 目录。
- WebQSP 测试样本 `WebQTest-863` 无路径及嵌入，原训练逻辑会跳过，
  因此其可用测试数为 1638。本次未修改该样本。

第一阶段每个数据集选取真实且候选图完整的 train/validation/test = 2/1/1 条。
CWQ 来自本地 Hugging Face Arrow 缓存；WebQSP 通过 Dataset Viewer 读取，
排除了 `truncated_cells` 不为空的记录。选择较小且有监督路径的图适合短测，
不代表全量数据分布。原始记录保存在各检索输出目录的 `raw_samples.json`。

## 已完成的训练检查

所有结果在仓库下 `outputs/core_smoke_20260922/`，与原权重分开。

| 项目 | WebQSP | CWQ |
| --- | --- | --- |
| 检索器真实优化步数 | 2 | 2 |
| 检索器平均 loss | 0.03350727 | 0.01542571 |
| 检索器检测层最大参数变化 | 0.00200121 | 0.00200135 |
| Llama3 映射器真实优化步数 | 2 | 2 |
| 映射器逐步 loss | 1.7188 / 0.3008 | 2.6250 / 0.1089 |
| 映射器逐步梯度范数 | 52.5829 / 26.6506 | 193.1537 / 92.8794 |
| 短测 validation loss | 12.3125 | 7.0703125 |
| 短测 test loss | 10.5625 | 7.265625 |

检索器调用生产代码的 `train_epoch`、`eval_epoch` 和 `infer_split`，
从原权重开始短测；保存并严格重载短测权重后，完成三个 split 的打分及路径恢复。
检索报告见 `retriever_webqsp/report.json` 和 `retriever_cwq/report.json`。

映射器通过正式 `scripts/train_deepspeed.sh` 启动，使用原 ZeRO-3、BF16、
最大长度 2600、最多 32 条路径、学习率 2e-3，冻结 Llama3。
仅将步数设为 2、各 split 样本上限设为 2、梯度累积设为 1。
两个任务均正常退出，完成验证、最终测试及 checkpoint 保存；每个最终
`mm_projector.bin` 包含 117 个有限值张量。两步训练的 loss 不构成收敛证据。

映射器证据：

- `webqsp_train_v3/smoke_train_metrics.json`、`cwq_train/smoke_train_metrics.json`。
- 对应目录下的 `eval_results/validation_history.jsonl` 和 `eval_results/test_final.json`。
- 对应目录下的 `checkpoint-2/` 和最终 `mm_projector.bin`。

## 推理与阶段衔接

原有 WebQSP、CWQ 的 Llama3 adapter 各推理两条原第二阶段测试数据：
均为 2/2 成功、无生成异常、无缺图跳过，输出均能按 JSON 解析。
见 `webqsp_existing/summary.json` 和 `cwq_existing/summary.json`。
生成明细又通过 `tools/extract_fields_to_json.py` 及带原别名资源的 `eval.py`，
两套数据均正常完成后处理与评分；这些微型样本分数不代表全量精度。

检索短测新恢复的路径已使用正式 `scripts/process_webqsp_e5.py` 重新编码，
两个数据集的三个 split 均通过 `--require-all-embeddings` 严格预检，
无缺失 ID、无格式异常。见 `retriever_*/bridge_preflight.json`。

本次保存的两个最终 adapter 均已重新加载，各接入一条上述新路径测试数据，
完成 Llama3 生成，`generation_error_count=0`、`skipped_missing_graph_count=0`。
结果在 `webqsp_bridge_new_adapter/` 和 `cwq_bridge_new_adapter/`。
这证明了保存加载及两个阶段的数据衔接，但新 adapter 仅训练两步，回答均不正确；
WebQSP 使用了纯文本解析回退，CWQ 输出也有无关内容，不应拿短测权重作正式推理模型。

另外，两个原已训练 adapter 也分别接入一条新恢复/编码的测试路径，
均完成生成及 JSON 解析，且无缺图跳过；结果在
`webqsp_bridge_original_adapter/` 和 `cwq_bridge_original_adapter/`。
这两条的答案没有命中当前 gold，因此这里也只判定程序链路通过，
不宣称精度达到论文结果。原 adapter、新 adapter 合计检查 8 次样本生成，
运行异常数为 0。

所有 GPU 任务结束后，GPU2 回到 42 MiB、0% 利用率。
改动的 Python AST 检查、Shell `bash -n` 和 `git diff --check` 均通过。
本次产物约 727 MiB，保留在上述独立输出目录，未提交到 git。

## 本次修复

1. DeepSpeed 0.14 对 `CUDA_VISIBLE_DEVICES=2` 的物理编号处理：
   启动器先枚举物理 GPU，再通过 `--include localhost:2` 限定 worker。
2. Accelerate 0.32 的优化器 `train()/eval()` 兼容：底层普通优化器无该方法时
   使用空操作，不改变参数更新算法；有实际模式钩子的优化器仍保留其方法。
3. 旧版 Llama FlashAttention monkey patch 不支持新 Transformers 的
   `cache_position`/RoPE 接口。检测到新版接口时使用 Transformers 原生注意力，
   保留旧版适用分支。没有替换模型方法或图适配器结构。
4. 增加可选的短测样本上限、步数参数和训练指标记录，默认不限制正式训练。
5. 整理后的脚本路径、数据目录传递、别名评测资源检查和数据预检已补齐；
   文档明确区分两阶段数据，前置模块不再属于核心链路。

## 实测环境

没有重新安装或升级现有环境：

- 映射器环境：Python 3.10，torch 2.1.0+cu118，
  transformers 4.48.3，accelerate 0.32.1，deepspeed 0.14.4，flash-attn 2.3.6。
- 检索与路径编码环境：torch 2.5.1+cu121，
  transformers 4.48.0，torch-geometric 2.7.0；用于检索与 ModernBERT 编码。

实测直接依赖另见 `requirements/tested_adapter.txt` 和 `requirements/tested_retrieval.txt`。
仓库旧 requirements 的版本不等同于上述实测环境，不能将本次结果视为
对全新安装环境的验证。DeepSpeed 的 libaio/CUTLASS/内核版本警告未阻止本次短测。

## 复现入口

映射器两步命令见 README；将数据放在 `data/<dataset>_may_merged`，
并设置 `LLAGA_LLAMA3_MODEL_BASE=./llama/Meta-Llama-3-8B-Instruct`。
每次指定一个新的 `LLAGA_OUTPUT_DIR`，避免覆盖其他实验。

检索短测在 `gte` 环境执行，示例（仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
TRANSFORMERS_CACHE="$PWD/.cache/huggingface/hub" \
python scripts/smoke_retriever.py --dataset webqsp \
  --output-dir outputs/core_smoke_20260922/retriever_webqsp \
  --checkpoint checkpoints/retriever/webqsp/cpt.pth \
  --identifiers retrieval/data_files/webqsp/entity_identifiers.txt
```

该命令复用已保存的原始短测样本，并更新其独立短测产物，不覆盖源 checkpoint。
CWQ 替换数据集、输出目录及 checkpoint，增加
`--hf-cache-root .cache/huggingface/datasets`。
CWQ checkpoint 为
`checkpoints/retriever/cwq/cpt.pth`。

路径编码在 `gte` 环境执行：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
python scripts/process_webqsp_e5.py --dataset webqsp \
  --base-dir outputs/core_smoke_20260922/retriever_webqsp/merged \
  --embedding-type Alibaba-NLP/gte-modernbert-base \
  --model-name-or-path Alibaba-NLP/gte-modernbert-base \
  --batch-size 32 --max-length 512
```

新 adapter 重载推理在 `llaga` 环境执行：

```bash
CUDA_VISIBLE_DEVICES=2 python eval/eval_webqsp_checkpoints.py \
  --dataset webqsp --data_dir outputs/core_smoke_20260922/retriever_webqsp/merged \
  --output_dir outputs/core_smoke_20260922/webqsp_train_v3 \
  --model_base ./llama/Meta-Llama-3-8B-Instruct \
  --conv_mode llama_3 --pretrained_embedding_type Alibaba-NLP/gte-modernbert-base \
  --splits test --checkpoints final --device cuda --device_map cuda:0 \
  --torch_dtype bfloat16 --max_graph_paths 32 --limit 1 --max_new_tokens 64 \
  --results_dir outputs/core_smoke_20260922/webqsp_bridge_new_adapter
```
