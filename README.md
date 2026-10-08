# GRLprof

GRLprof 是面向复杂知识图谱问答的结构感知检索与推理框架。系统对候选图中的三元组打分，恢复推理路径，将路径结构映射为软提示，辅助冻结的本地大语言模型生成答案。

核心输入是**已包含主题实体和候选图的 KGQA 数据**。实体链接和在线 Freebase 子图获取不是核心流程；相关实验代码保留在 `grlprof/entity_linking/` 和 `grlprof/freebase_pipeline/`，默认不调用。

## 1. 方法与验证范围

```text
问题 + 主题实体 + 候选知识图谱
  → 三元组相关性打分 → 高分路径恢复
  → 路径语义编码 → 结构感知知识映射器
  → 软提示注入 → 冻结的 Llama3 生成答案
```

| 阶段 | 实测配置 | 作用 |
| --- | --- | --- |
| 检索语义编码 | Alibaba-NLP/gte-large-en-v1.5，1024 维 | 问题、实体、关系表示 |
| 路径编码 | Alibaba-NLP/gte-modernbert-base，768 维 | 保留三元组角色与路径结构 |
| 知识映射器 | 八层路径 Transformer 及投影模块 | 映射到语言模型嵌入空间 |
| 答案生成 | Meta-Llama-3-8B-Instruct，冻结基座 | 基于问题与图软提示生成答案 |

WebQSP、CWQ 均已通过真实数据的两步检索训练、两步映射器训练、权重保存重载、跨阶段生成及评分后处理。没有执行完整重训，不能据此宣称收敛或复现论文指标。**全新环境安装尚未独立验证**；下面记录的是实测环境的直接依赖版本，不是完整锁文件。Llama2、OPT 和可选前置模块不在本轮验证范围内。

详见 [方法说明](docs/method.md)、[数据结构](docs/data.md) 和 [短测报告](docs/core_smoke_20260922.md)。报告路径已替换为可移植示例；本机日志、数据和权重不随源码发布。

## 2. 项目目录

```text
retrieval/       检索器训练、推理及语义缓存
model/          图知识映射器及语言模型接口
train/          映射器训练入口
eval/           checkpoint 推理及交互入口
scripts/        路径恢复、编码、预检和流水线
tools/          答案后处理、源码发布检查
configs/        DeepSpeed 与路径配置示例
requirements/   分阶段依赖及实测版本记录
docs/           方法、数据、验证及发布说明
data/           本地数据（不提交）
checkpoints/    本地权重（不提交）
llama/          本地 Llama 基座（不提交）
```

部分内部类名、checkpoint model type 和环境变量保留 `llaga` / `LLAGA_*` 命名，以兼容已有权重；项目对外名称为 GRLprof。

## 3. 环境准备

需要 Linux、Python 3.10、兼容的 NVIDIA 驱动及 CUDA GPU。安装编译型依赖还需要匹配的 CUDA toolkit 和编译工具。模型为 8B 级别，请根据显存调节批量、序列长度、路径数及 DeepSpeed 配置；不保证任意显卡均可运行。

从仓库根目录执行。建议隔离两套环境，不要把历史 `requirements/adapter.txt` 与实测版本混装。

### 映射器训练与 Llama3 推理

```bash
conda create -n grlprof-adapter python=3.10 -y
conda activate grlprof-adapter
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/tested_adapter.txt
```

实测环境还安装了 `flash-attn==2.3.6`。新 Transformers 可使用原生注意力；下面的训练命令设置 `LLAGA_DISABLE_FLASH_ATTN=1` 直接走 `train/train.py`。若使用旧入口，则需自行安装与 CUDA 匹配的 FlashAttention。

### 检索与路径编码

```bash
conda create -n grlprof-retrieval python=3.10 -y
conda activate grlprof-retrieval
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/tested_retrieval.txt
python -m pip install xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu121
```

GTE-large 使用 `trust_remote_code=True`；请审阅并信任模型仓库代码后再使用。首次下载需要联网，缓存完备后可设置 `HF_HUB_OFFLINE=1`。

### 本地配置

```bash
cp configs/model_paths.example.env .env
# 编辑 .env，填入自己的模型与 adapter 路径，再显式加载：
set -a
source .env
set +a
export HF_HOME="$PWD/.cache/huggingface"
export WANDB_MODE=disabled
```

脚本不会自动读取 `.env`。选择空闲 GPU 时设置 `CUDA_VISIBLE_DEVICES=0`，编号以自己的机器为准。单卡环境中的 `cuda:0` 表示可见 GPU，不一定是物理 GPU0。

## 4. 必须准备的外部资源

源码仓库不分发数据、模型或训练权重，目前也没有公开 checkpoint 下载链接。

| 资源 | 获取 / 放置方式 |
| --- | --- |
| 第一阶段原始数据 | `retrieval/emb.py` 从 Hugging Face 加载 `ml1996/webqsp` 或 `rmanluo/RoG-cwq` |
| 非文本实体 ID 表 | `retrieval/data_files/<dataset>/entity_identifiers.txt`；需与现有权重训练时一致 |
| 两个语义编码模型 | 首次按模型 ID 下载，或提前缓存 |
| Llama3 基座 | 自行取得访问许可，放在 `llama/Meta-Llama-3-8B-Instruct` 或设置 `LLAGA_LLAMA3_MODEL_BASE` |
| 检索器 checkpoint | 自行训练得到 `cpt.pth`，或使用兼容权重 |
| 映射器 checkpoint | 第二阶段生成 `config.json` 和 `mm_projector.bin` |
| 别名评测资源 | 自行准备并放在 `data/aliases/`，文件名见评测章节 |

缺少实体 ID 表时，当前 `retrieval/emb.py` 会警告并按空集合继续，这会改变非文本实体处理，**不能认为与已有权重的原配置等价**。不要用这个降级流程宣称复现原实验。

数据和基础模型分别遵循各自许可及访问条款。没有外部资源时，不能仅凭克隆代码完成全链路运行。

## 5. 第一阶段：检索训练与推理

激活 `grlprof-retrieval`，先准备实体 ID 表，再从 `retrieval/` 执行：

```bash
cd retrieval
CUDA_VISIBLE_DEVICES=0 python emb.py -d webqsp
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python train.py -d webqsp
```

训练生成带时间戳的运行目录。将 `RUN_DIR` 替换为实际目录名：

```bash
RUN_DIR=webqsp_YOUR_RUN_ID
CUDA_VISIBLE_DEVICES=0 python inference.py -p "$RUN_DIR/cpt.pth"
python eval.py -d webqsp -p "$RUN_DIR/retrieval_result.pth"
cd ..
```

输出为 `train.json`、`val.json`、`test.json` 的高分三元组及供检索评测使用的 `retrieval_result.pth`。CWQ 同时替换数据集参数和目录。

## 6. 两阶段之间：路径恢复与编码

在仓库根目录、`grlprof-retrieval` 环境运行。`RUN_DIR` 沿用上一步名称：

```bash
python scripts/expand_webqsp_scored_triples.py \
  --dataset webqsp --input-root "retrieval/$RUN_DIR" \
  --output-root data/webqsp_may_merged \
  --hf-cache-root "$HF_HOME/datasets"

CUDA_VISIBLE_DEVICES=0 python scripts/process_webqsp_e5.py \
  --dataset webqsp --base-dir data/webqsp_may_merged \
  --embedding-type Alibaba-NLP/gte-modernbert-base --batch-size 32

python scripts/preflight_core_pipeline.py \
  --dataset webqsp --data-dir data/webqsp_may_merged \
  --embedding-type Alibaba-NLP/gte-modernbert-base
```

路径恢复读取第一阶段下载形成的本地 Arrow 缓存；不能把 merged 数据当作候选图输入。`process_webqsp_e5.py` 保留历史文件名，实际编码器由 `--embedding-type` 指定，并非固定使用 E5。

第二阶段布局：

```text
data/webqsp_may_merged/
  train.json
  validation.json
  test.json
  webqsp_paths_gte_modernbert_base_train.pt
  webqsp_paths_gte_modernbert_base_validation.pt
  webqsp_paths_gte_modernbert_base_test.pt
```

CWQ 使用 `data/cwq_may_merged/` 和 `cwq_paths_...pt`。JSON 含 `reasoning_paths`；缓存按 ID 保存路径内三元组的 768 维表示。完整字段见 [数据说明](docs/data.md)。

## 7. 第二阶段：映射器训练

切换 `grlprof-adapter`。先短测，再考虑正式训练；每次使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 LLAGA_DISABLE_FLASH_ATTN=1 \
LLAGA_MAX_STEPS=2 LLAGA_MAX_SAMPLES_PER_SPLIT=2 \
LLAGA_GRAD_ACCUM=1 LLAGA_DATALOADER_NUM_WORKERS=0 \
LLAGA_LOGGING_STEPS=1 LLAGA_SAVE_STEPS=2 \
LLAGA_OUTPUT_DIR=./outputs/smoke_webqsp \
./scripts/train_deepspeed.sh \
  llama3 QA webqsp 1 Alibaba-NLP/gte-modernbert-base data/webqsp_may_merged
```

短测应生成 `smoke_train_metrics.json`、`mm_projector.bin`、`config.json` 和 `eval_results/`。检查实际步数、有限 loss / 梯度和退出状态。样本上限在正常数据构建后生效，仍加载完整嵌入缓存。两步权重不适合正式推理。

正式训练示例（本轮未执行完整训练）：

```bash
CUDA_VISIBLE_DEVICES=0 LLAGA_DISABLE_FLASH_ATTN=1 \
LLAGA_OUTPUT_DIR=./checkpoints/webqsp/grlprof-llama3 \
./scripts/train_deepspeed.sh \
  llama3 QA webqsp 1 Alibaba-NLP/gte-modernbert-base data/webqsp_may_merged
```

常用变量：`LLAGA_LLAMA3_MODEL_BASE`、`LLAGA_OUTPUT_DIR`、`LLAGA_MAX_LEN`（默认 2600）、`LLAGA_MAX_GRAPH_PATHS`（默认 32）、`LLAGA_GRAD_ACCUM`、`LLAGA_LR`、`LLAGA_EPOCHS`。`LLAGA_DRY_RUN=1` 只检查展开后的命令。

## 8. 推理、答案提取与评测

使用已训练 adapter；仅检查重载时可换成短测目录。通过 `--results_dir` 指定独立结果目录：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_webqsp_checkpoints.py \
  --dataset webqsp --data_dir data/webqsp_may_merged \
  --output_dir checkpoints/webqsp/grlprof-llama3 \
  --model_base "$LLAGA_LLAMA3_MODEL_BASE" --conv_mode llama_3 \
  --pretrained_embedding_type Alibaba-NLP/gte-modernbert-base \
  --splits test --checkpoints final --torch_dtype bfloat16 \
  --device cuda --device_map cuda:0 --max_graph_paths 32 \
  --limit 2 --max_new_tokens 64 --results_dir outputs/webqsp_eval

python tools/extract_fields_to_json.py outputs/webqsp_eval/final_test_details.jsonl
python eval.py --dataset webqsp \
  --input_file outputs/webqsp_eval/final_test_details_extracted.json \
  --alias_dir data/aliases --quiet
```

去掉 `--limit 2` 才评测整个 split；正式生成长度应按实验设置，不要直接沿用短测的 64。`summary.json` 记录统计，`*_details.jsonl` 保存逐条结果。推理内置指标与带别名的 `eval.py` 口径不同，论文对比必须统一口径。

别名评分需要以下本地资源，缺失时会报错：

- `WQSP_aliase_data.json`
- `CWQ_aliase_data31158.json`
- `ComplexWebQuestions_test_wans.json`
- `cwq_aname_dict.json`

也可设置 `GRLPROF_ALIAS_DIR`。这些资源没有随仓库分发，配套发布前需确认来源和再分发许可。

已准备好第二阶段数据后，`scripts/train_infer_extract_eval_pipeline.sh` 可执行训练 → 推理 → 提取 → 评分；它**不包含**第一阶段检索与路径编码。运行前阅读脚本顶部参数说明。

## 9. 常见问题与可选功能

- 找不到模型：检查 `.env` 是否显式加载、本地路径是否存在，或是否取得模型访问许可。
- 找不到嵌入：先预检数据集前缀、文件名和 ID。无路径样本可能按原逻辑跳过，必须报告有效样本数。
- 显存不足：降低 batch size、`LLAGA_MAX_LEN`、`LLAGA_MAX_GRAPH_PATHS` 或调整 ZeRO；在实验中记录这些变化。
- 旧注意力报错：使用实测 Transformers 版本和原生入口，不要强行套用旧 FlashAttention 补丁。
- GPU 编号错误：脚本会把数字形式的 `CUDA_VISIBLE_DEVICES` 转为 DeepSpeed 物理卡选择，不要再传入冲突的 include 配置。
- 资源不齐：可以运行 CPU 发布检查，但语法通过不等于模型链路已验证。

扩展与基线见 [实体链接](grlprof/entity_linking/README.md)、[Freebase 前置流程](grlprof/freebase_pipeline/README.md)、[文本推理基线](reasoning/README.md)。它们不是运行核心方法的前提。

## 10. 发布、来源与许可

```bash
python tools/check_release.py
python -m unittest discover -s tests -v
python tools/export_source.py --output outputs/release/GRLprof-source.zip
```

发布检查覆盖候选源码文件中的常见密钥、私人绝对路径、意外数据 / 权重、符号链接及可达历史中的敏感内容；它不能替代安全审计。源码包不含 `.git`，不带出旧历史中的作者邮箱。不要直接打包整个工作目录。

见 [发布清单](docs/release_checklist.md)。对外以独立 GRLprof 方法介绍；第三方代码来源、许可证及兼容性说明保留在 [NOTICE.md](NOTICE.md) 和 [licenses/](licenses/)，不因文案调整删除。
