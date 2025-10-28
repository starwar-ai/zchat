#!/bin/bash

# 针对4个T4 GPU的完整训练流程脚本
# 基于run1000.sh修改，专门为T4 GPU的16GB显存限制进行优化

set -e  # 遇到错误时退出

echo "🚀 开始T4 GPU训练流程..."

# 环境设置
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=".cache/nanochat"
export DATA_DIR="./data"
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $DATA_DIR

# 检查并安装uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置虚拟环境
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# 设置wandb运行名称
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="t4_training_$(date +%Y%m%d_%H%M%S)"
fi
echo "📊 Wandb运行名称: $WANDB_RUN"

# 重置报告
python -m nanochat.report reset

# 安装Rust和编译tokenizer
echo "🔧 编译 Rust tokenizer..."
if [ ! -d "$HOME/.cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "📊 开始数据准备..."
echo "=================================================="

# 步骤1: 下载并验证所有数据
echo "📥 下载训练和评估数据..."
python -m scripts.prepare_data --data-dir $DATA_DIR --num-base-shards 100 --num-workers 4

# 步骤2: 验证数据完整性
echo ""
echo "🔍 验证数据完整性..."
python -m nanochat.data_checker
if [ $? -ne 0 ]; then
    echo "❌ 数据完整性检查失败！请检查数据下载。"
    exit 1
fi

echo ""
echo "✅ 数据准备完成！"
echo "=================================================="
echo ""

# 训练tokenizer
echo "🔤 训练tokenizer..."
python -m scripts.tok_train --max_chars=2000000000  # 减少字符数以适应T4
python -m scripts.tok_eval

echo "🏋️ 开始基础模型训练..."

# 基础模型训练 - 针对T4优化
echo "📈 运行基础训练 (深度12, 批次大小4)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train -- --run=$WANDB_RUN

echo "📊 运行基础损失评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_loss

echo "📊 运行基础模型评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval

echo "🎯 开始中期训练..."

# 中期训练 - 针对T4优化
echo "📈 运行中期训练 (批次大小2)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_mid_train -- --run=$WANDB_RUN

echo "📊 运行中期训练评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i mid

echo "💬 开始SFT训练..."

# SFT训练 - 针对T4优化
echo "📈 运行SFT训练 (批次大小1)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_chat_sft -- --run=$WANDB_RUN

echo "📊 运行SFT评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft

echo "📋 生成最终报告..."
python -m nanochat.report generate

echo "🎉 T4训练流程完成！"
echo "📊 查看报告: python -m nanochat.report show"
echo "💬 启动聊天界面: python -m scripts.chat_web"

# 显示GPU使用情况
echo "🔍 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "✅ 所有训练步骤已完成！"
