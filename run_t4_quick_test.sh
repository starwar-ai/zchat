#!/bin/bash

# T4 GPU快速测试脚本
# 用于验证T4配置是否正常工作，运行较少的训练步数

set -e

echo "🧪 开始T4 GPU快速测试..."

# 环境设置
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=".cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 检查并安装uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置虚拟环境
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# 设置wandb运行名称
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="t4_quick_test_$(date +%Y%m%d_%H%M%S)"
fi
echo "📊 Wandb运行名称: $WANDB_RUN"

# 重置报告
python -m nanochat.report reset

# 安装Rust和编译tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 下载评估数据
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "📥 下载评估数据包..."
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# 下载身份对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "📊 开始数据准备..."

# 训练tokenizer - 使用最少的数据
echo "🔤 训练tokenizer..."
python -m nanochat.dataset -n 2  # 最少数据量
python -m scripts.tok_train --max_chars=100000000  # 最少字符数
python -m scripts.tok_eval

echo "🏋️ 开始基础模型训练 (快速测试)..."

# 基础模型训练 - 快速测试版本
echo "📈 运行基础训练 (深度8, 批次大小2, 100步)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_train -- --run=$WANDB_RUN --depth=8 --device_batch_size=2 --num_iterations=100

echo "📊 运行基础损失评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_loss

echo "📊 运行基础模型评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval

echo "🎯 开始中期训练 (快速测试)..."

# 中期训练 - 快速测试版本
echo "📈 运行中期训练 (批次大小1, 50步)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_mid_train -- --run=$WANDB_RUN --device_batch_size=1 --num_iterations=50

echo "📊 运行中期训练评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i mid

echo "💬 开始SFT训练 (快速测试)..."

# SFT训练 - 快速测试版本
echo "📈 运行SFT训练 (批次大小1, 20步)..."
torchrun --standalone --nproc_per_node=4 -m scripts.t4_chat_sft -- --run=$WANDB_RUN --device_batch_size=1 --num_iterations=20

echo "📊 运行SFT评估..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft

echo "📋 生成最终报告..."
python -m nanochat.report generate

echo "🎉 T4快速测试完成！"
echo "📊 查看报告: python -m nanochat.report show"
echo "💬 启动聊天界面: python -m scripts.chat_web"

# 显示GPU使用情况
echo "🔍 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "✅ 快速测试已完成！"
