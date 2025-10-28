# 数据管理说明

本项目已更新为统一的数据管理流程。所有训练和评估数据都会集中下载到 `data/` 目录，并在训练开始前进行完整性检查。

## 目录结构

```
data/
├── base_data/                    # FineWeb-Edu 基础训练数据 (parquet shards)
├── eval_bundle/                  # 评估数据包
├── identity_conversations.jsonl  # 身份对话数据
├── smoltalk/                     # SmolTalk 数据集 (train.parquet, test.parquet)
├── mmlu/                         # MMLU 数据集 (多个 parquet 文件)
├── humaneval/                    # HumanEval 数据集
├── gsm8k/                        # GSM8K 数据集
└── arc/                          # ARC 数据集
```

## 使用方法

### 1. 下载所有数据

使用统一的数据准备脚本下载所有必需的数据：

```bash
# 下载所有数据（包括评估数据、HuggingFace 数据集、基础训练数据）
python -m scripts.prepare_data --data-dir ./data

# 只下载部分基础训练数据 shards（例如 100 个，用于快速测试）
python -m scripts.prepare_data --data-dir ./data --num-base-shards 100

# 使用多进程加速下载
python -m scripts.prepare_data --data-dir ./data --num-workers 8
```

### 2. 检查数据完整性

可以单独检查数据完整性而不下载：

```bash
# 仅检查数据完整性
python -m scripts.prepare_data --data-dir ./data --check-only

# 或使用数据检查模块
python -m nanochat.data_checker
```

### 3. 选择性下载

可以跳过某些类型的数据下载：

```bash
# 跳过 S3 文件下载
python -m scripts.prepare_data --skip-s3

# 跳过 HuggingFace 数据集下载
python -m scripts.prepare_data --skip-hf

# 跳过基础训练数据下载
python -m scripts.prepare_data --skip-base

# 强制重新下载已存在的文件
python -m scripts.prepare_data --force
```

## 训练流程

### 自动数据检查

所有训练脚本在开始训练前会自动检查数据完整性：

- `scripts/tok_train.py` - Tokenizer 训练（需要基础训练数据）
- `scripts/t4_train.py` - 基础模型训练（需要基础训练数据）
- `scripts/t4_mid_train.py` - 中期训练（需要 HF 数据集和 identity_conversations.jsonl）
- `scripts/t4_chat_sft.py` - SFT 训练（需要 HF 数据集和 identity_conversations.jsonl）

如果数据不完整，训练会自动停止并提示需要下载哪些数据。

### 完整训练流程

使用更新后的训练脚本：

```bash
# 运行完整的 T4 训练流程（自动下载和检查数据）
bash run_t4_training.sh
```

## 数据来源

### S3 文件
- **eval_bundle.zip** - 评估数据包
  - 来源: https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip

- **identity_conversations.jsonl** - 身份对话数据
  - 来源: https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

### HuggingFace 数据集
- **SmolTalk** - 对话数据集
  - Repo: HuggingFaceTB/smol-smoltalk
  - Splits: train, test

- **MMLU** - 多任务语言理解
  - Repo: cais/mmlu
  - Configs: all, auxiliary_train
  - Splits: train, validation, dev, test

- **HumanEval** - 代码评估
  - Repo: openai/openai_humaneval
  - Splits: test

- **GSM8K** - 数学问题
  - Repo: openai/gsm8k
  - Configs: main, socratic
  - Splits: train, test

- **ARC** - AI2 推理挑战
  - Repo: allenai/ai2_arc
  - Configs: ARC-Easy, ARC-Challenge
  - Splits: train, validation, test

### 基础训练数据
- **FineWeb-Edu** - 预训练文本数据
  - 来源: ModelScope (Thackeray/karpathy-fineweb-edu-100b-shuffle-240shard)
  - 总共 1823 个 shard 文件 (shard_00000.parquet - shard_01822.parquet)

## 常见问题

### Q: 数据下载失败怎么办？

A: 脚本内置了重试机制（最多 5 次，指数退避）。如果仍然失败：
1. 检查网络连接
2. 重新运行下载命令（已下载的文件会被跳过）
3. 使用 `--num-workers 1` 减少并发数

### Q: 磁盘空间不足怎么办？

A:
1. 可以只下载部分基础训练数据：`--num-base-shards 50`
2. 跳过不需要的数据类型：`--skip-base` 或 `--skip-hf`

### Q: 如何清理和重新下载数据？

A:
```bash
# 删除数据目录
rm -rf data/

# 重新下载
python -m scripts.prepare_data --data-dir ./data
```

### Q: 训练脚本如何知道数据在哪里？

A: 训练脚本会优先使用 `./data/` 目录。如果该目录不存在，会回退到 `.cache/nanochat/` 目录（旧的位置）。

## 改进说明

与之前的版本相比，新的数据管理系统有以下改进：

1. **集中化管理** - 所有数据集中在 `data/` 目录
2. **统一下载** - 一个命令下载所有数据
3. **完整性检查** - 训练前自动验证数据完整性
4. **更好的错误处理** - 明确提示缺失的数据项
5. **并行下载** - 支持多进程加速下载
6. **增量下载** - 自动跳过已存在的文件
