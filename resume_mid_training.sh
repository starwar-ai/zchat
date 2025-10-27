#!/bin/bash

# Resume training from mid-training phase (skipping base pretraining)
# This assumes base training has already completed and base checkpoints exist.

# Set environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

echo "=========================================="
echo "Resume Training - Starting from Mid-Training"
echo "=========================================="
echo ""
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo ""

# Check if base checkpoints exist
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    echo "ERROR: Base checkpoints not found at $NANOCHAT_BASE_DIR/base_checkpoints"
    echo ""
    echo "Please ensure:"
    echo "  1. Base training has completed successfully"
    echo "  2. NANOCHAT_BASE_DIR points to the correct directory"
    echo "  3. Or set NANOCHAT_BASE_DIR environment variable to the checkpoint location"
    echo ""
    echo "Example:"
    echo "  export NANOCHAT_BASE_DIR=/path/to/checkpoints"
    echo "  bash resume_mid_training.sh"
    exit 1
fi

echo "✓ Found base checkpoints at: $NANOCHAT_BASE_DIR/base_checkpoints"
ls -lh $NANOCHAT_BASE_DIR/base_checkpoints/
echo ""

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Please run:"
    echo "  uv venv && uv sync --extra gpu"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated virtual environment"
echo ""

# Wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
    echo "Using dummy wandb run (no logging). Set WANDB_RUN to enable wandb logging."
else
    echo "Using wandb run: $WANDB_RUN"
fi
echo ""

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)
echo "=========================================="
echo "Phase 1: Mid-Training"
echo "=========================================="

# Download identity conversations if not present
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity conversations..."
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    echo "✓ Downloaded identity conversations"
else
    echo "✓ Identity conversations already exist"
fi
echo ""

# Run midtraining
echo "Starting mid-training..."
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
echo "✓ Mid-training completed"
echo ""

# Evaluate the mid-trained model
echo "Evaluating mid-trained model..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
echo "✓ Mid-training evaluation completed"
echo ""

# -----------------------------------------------------------------------------
# Supervised Finetuning
echo "=========================================="
echo "Phase 2: Supervised Fine-Tuning (SFT)"
echo "=========================================="

# Run SFT
echo "Starting SFT..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
echo "✓ SFT completed"
echo ""

# Evaluate SFT model
echo "Evaluating SFT model..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
echo "✓ SFT evaluation completed"
echo ""

# -----------------------------------------------------------------------------
# Generate report
echo "=========================================="
echo "Generating Report"
echo "=========================================="
python -m nanochat.report generate
echo "✓ Report generated"
echo ""

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. View the report: cat report.md"
echo "  2. Chat with your model (CLI): python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo "  3. Chat with your model (Web UI): python -m scripts.chat_web"
echo ""
echo "Optional: Run Reinforcement Learning"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K"
echo ""
