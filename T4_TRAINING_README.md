# T4 GPU 训练指南

本文档说明如何在4个T4 GPU的服务器上运行nanochat训练。

## 硬件要求

- 4个NVIDIA T4 GPU (每个16GB显存)
- 至少64GB系统内存
- 足够的存储空间用于数据集和模型检查点

## 文件说明

### 训练脚本

1. **`scripts/t4_train.py`** - 针对T4优化的基础模型训练脚本
2. **`scripts/t4_mid_train.py`** - 针对T4优化的中期训练脚本  
3. **`scripts/t4_chat_sft.py`** - 针对T4优化的SFT训练脚本

### 启动脚本

1. **`run_t4_training.sh`** - 完整的T4训练流程
2. **`run_t4_quick_test.sh`** - 快速测试脚本，用于验证配置

## T4优化配置

### 基础模型训练 (t4_train.py)

- **模型深度**: 12层 (原来20-32层)
- **序列长度**: 1024 (原来2048)
- **设备批次大小**: 4 (原来32)
- **总批次大小**: 131,072 tokens (原来524,288)
- **评估频率**: 每100步 (原来250步)

### 中期训练 (t4_mid_train.py)

- **设备批次大小**: 2 (原来32)
- **总批次大小**: 65,536 tokens (原来524,288)
- **评估频率**: 每75步 (原来150步)

### SFT训练 (t4_chat_sft.py)

- **设备批次大小**: 1 (原来4)
- **目标样本数**: 8 (原来32)
- **评估频率**: 每50步 (原来100步)

## 使用方法

### 1. 快速测试

首先运行快速测试来验证配置：

```bash
./run_t4_quick_test.sh
```

这将运行：
- 深度8的模型
- 100步基础训练
- 50步中期训练  
- 20步SFT训练

### 2. 完整训练

如果快速测试成功，运行完整训练：

```bash
./run_t4_training.sh
```

这将运行：
- 深度12的模型
- 完整的基础训练
- 完整的中期训练
- 完整的SFT训练

### 3. 单独运行训练步骤

你也可以单独运行每个训练步骤：

```bash
# 基础训练
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train

# 中期训练
torchrun --standalone --nproc_per_node=4 -m scripts.t4_mid_train

# SFT训练
torchrun --standalone --nproc_per_node=4 -m scripts.t4_chat_sft
```

## 参数调整

如果遇到显存不足的问题，可以进一步调整参数：

### 减少批次大小
```bash
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train -- --device_batch_size=2
```

### 减少模型深度
```bash
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train -- --depth=8
```

### 减少序列长度
```bash
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train -- --max_seq_len=512
```

## 监控训练

### 查看GPU状态
```bash
nvidia-smi
```

### 查看训练日志
训练过程中会显示：
- 损失值
- 学习率
- 每步时间
- 吞吐量 (tokens/sec)
- 模型利用率 (MFU)

### 查看Wandb日志
如果设置了Wandb，可以在Wandb界面查看详细的训练指标。

## 故障排除

### 显存不足 (OOM)
1. 减少 `device_batch_size`
2. 减少 `max_seq_len`
3. 减少 `depth`

### 训练速度慢
1. 检查GPU利用率 (`nvidia-smi`)
2. 确保数据加载不是瓶颈
3. 考虑减少 `eval_every` 频率

### 分布式训练问题
1. 确保所有4个GPU都可用
2. 检查网络连接
3. 确保端口没有被占用

## 预期性能

在4个T4 GPU上：
- **基础训练**: 约2-4小时 (取决于模型大小)
- **中期训练**: 约1-2小时
- **SFT训练**: 约30分钟-1小时

## 输出文件

训练完成后，模型检查点将保存在：
- 基础模型: `~/.cache/nanochat/checkpoints/`
- 中期模型: `~/.cache/nanochat/mid_checkpoints/`
- SFT模型: `~/.cache/nanochat/chatsft_checkpoints/`

## 注意事项

1. T4 GPU的显存限制意味着需要使用较小的批次大小
2. 模型深度和序列长度需要相应调整
3. 训练时间会比H100等高端GPU更长
4. 建议先运行快速测试验证配置

## 支持

如果遇到问题，请检查：
1. GPU驱动和CUDA版本
2. PyTorch版本兼容性
3. 显存使用情况
4. 训练日志中的错误信息
