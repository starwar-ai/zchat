# 本地数据集使用指南

本项目已经支持从本地加载 HuggingFace 数据集，避免每次运行都需要从网上下载数据。

## 功能概述

- **在线模式**：默认行为，从 HuggingFace 下载数据集
- **本地模式**：从本地 parquet 文件加载数据集，支持离线使用

## 支持的数据集

- SmolTalk (smoltalk)
- MMLU (mmlu)
- HumanEval (humaneval)
- GSM8K (gsm8k)
- ARC (arc)

## 使用方法

### 1. 下载数据集

首先运行下载脚本，将所有数据集下载到本地：

```bash
python scripts/download_datasets.py
```

或者只下载特定数据集：

```bash
python scripts/download_datasets.py --dataset smoltalk
```

### 2. 在代码中使用本地数据集

在创建任务实例时，添加 `data_dir` 参数：

```python
from tasks.smoltalk import SmolTalk

# 使用本地数据集
task = SmolTalk(split='train', data_dir='./data')

# 或者继续使用在线模式（默认行为）
task = SmolTalk(split='train')
```

### 3. 数据集存储结构

数据集存储在 `./data/` 目录下：

```
data/
├── smoltalk/
│   ├── train.parquet
│   └── test.parquet
├── mmlu/
│   ├── all_train.parquet
│   ├── all_validation.parquet
│   ├── all_dev.parquet
│   ├── all_test.parquet
│   └── auxiliary_train_train.parquet
├── humaneval/
│   └── test.parquet
├── gsm8k/
│   ├── main_train.parquet
│   ├── main_test.parquet
│   ├── socratic_train.parquet
│   └── socratic_test.parquet
└── arc/
    ├── ARC-Easy_train.parquet
    ├── ARC-Easy_validation.parquet
    ├── ARC-Easy_test.parquet
    ├── ARC-Challenge_train.parquet
    ├── ARC-Challenge_validation.parquet
    └── ARC-Challenge_test.parquet
```

## 修改的任务类

所有任务类都已修改，支持 `data_dir` 参数：

- `SmolTalk(split, data_dir=None, **kwargs)`
- `MMLU(subset, split, data_dir=None, **kwargs)`
- `HumanEval(data_dir=None, **kwargs)`
- `GSM8K(subset, split, data_dir=None, **kwargs)`
- `ARC(subset, split, data_dir=None, **kwargs)`

## 注意事项

1. 如果指定了 `data_dir` 但本地数据集文件不存在，程序会报错并提示运行下载脚本
2. 本地数据集使用 parquet 格式，比原始 HuggingFace 数据集加载更快
3. 数据集在下载时会自动 shuffle，确保结果可重现

## 测试

运行测试脚本验证修改是否正确：

```bash
python test_local_loading.py
```
