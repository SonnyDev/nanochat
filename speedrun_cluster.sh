#!/bin/bash
# =============================================================================
# nanochat speedrun adapted for HPC cluster (ROCm GPUs, SLURM, no internet)
# =============================================================================
# Usage:
#   1. FROM LOGIN NODE (has internet): run the prep step first:
#      bash nanochat/speedrun_cluster.sh --prep
#   2. Then submit the training job:
#      sbatch nanochat/speedrun_cluster.sh
# =============================================================================

# --- PREP MODE: run from login node to download data + install deps ----------
if [ "$1" = "--prep" ]; then
    echo "=== PREP MODE (run from login node with internet) ==="
    cd "$(dirname "$0")"

    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
    mkdir -p "$NANOCHAT_BASE_DIR"

    # Use the shared venv that already has PyTorch ROCm
    VENV_PATH="/lus/work/CT10/c1916309/smuhima/project_ocs/new_venv"
    source "$VENV_PATH/bin/activate"

    # Install nanochat Python deps (skip torch, already in venv)
    pip install datasets fastapi matplotlib psutil regex rustbpe scipy \
        tabulate tiktoken tokenizers uvicorn wandb --quiet

    # Download data shards (8 for tokenizer, 240 for pretraining)
    echo "Downloading data shards..."
    python -m nanochat.dataset -n 240

    # Download identity conversations for midtraining
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    echo "=== PREP DONE. Now submit: sbatch nanochat/speedrun_cluster.sh ==="
    exit 0
fi

# --- SLURM HEADER -----------------------------------------------------------
#SBATCH --job-name=nanochat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=nanochat_%j.log
#SBATCH --error=nanochat_%j.err
# Adapt partition name to your cluster:
#SBATCH --partition=mi250
#SBATCH --account=c1916309

# --- ENV SETUP ---------------------------------------------------------------
set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES

mkdir -p "$NANOCHAT_BASE_DIR"

# Use the shared venv with PyTorch ROCm
VENV_PATH="/lus/work/CT10/c1916309/smuhima/project_ocs/new_venv"
source "$VENV_PATH/bin/activate"

# cd into nanochat dir
cd "$(dirname "$0")"

NPROC_PER_NODE=8
WANDB_RUN=${WANDB_RUN:-dummy}

# --- REPORT ------------------------------------------------------------------
echo "=== Starting nanochat training ==="
python -m nanochat.report reset

# --- TOKENIZER ---------------------------------------------------------------
echo "=== Training tokenizer ==="
# Data should already be downloaded from --prep step
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536
python -m scripts.tok_eval

# --- BASE MODEL (pretraining) -----------------------------------------------
echo "=== Pretraining base model (d20, ~561M params) ==="
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.base_train -- \
    --depth=20 \
    --target_param_data_ratio=20 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# --- MIDTRAINING -------------------------------------------------------------
echo "=== Midtraining ==="
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval -- -i mid

# --- SFT ---------------------------------------------------------------------
echo "=== Supervised Finetuning ==="
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval -- -i sft

# --- RL (optional) -----------------------------------------------------------
# echo "=== Reinforcement Learning ==="
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
#     -m scripts.chat_rl -- --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
#     -m scripts.chat_eval -- -i rl -a GSM8K

# --- REPORT ------------------------------------------------------------------
python -m nanochat.report generate
echo "=== Done! Report in report.md ==="
