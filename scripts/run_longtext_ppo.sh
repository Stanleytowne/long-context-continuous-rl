#!/bin/bash
################################################################################
# Long-Text PPO Training Runner
#
# This script runs PPO training for long-text continuous learning.
# The model learns to generate and answer questions about a long text document.
#
# Usage:
#   bash scripts/run_longtext_ppo.sh [long_text_path] [model_path] [output_dir]
#
# Example:
#   # Start from warmup checkpoint
#   bash scripts/run_longtext_ppo.sh \
#       /path/to/document.txt \
#       ./output/warmup \
#       ./output/ppo
#
#   # Start from base model
#   bash scripts/run_longtext_ppo.sh \
#       /path/to/document.txt \
#       Qwen/Qwen2.5-7B \
#       ./output/ppo
################################################################################

# Exit on error
set -e

# Default parameters
LONG_TEXT_PATH="${1:-data/sample_document.txt}"
MODEL_PATH="${2:-./outputs/warmup/global_step_15}"
OUTPUT_DIR="${3:-./outputs/longtext_ppo}"
NUM_GPUS="${4:-4}"

# Training parameters
TOTAL_EPOCHS=50
TRAIN_BATCH_SIZE=8           # Increased to ensure proper batch division (8*2/4=4)
ROLLOUT_BATCH_SIZE=16
LEARNING_RATE=1e-6
MAX_PROMPT_LENGTH=8096
MAX_RESPONSE_LENGTH=4096
PPO_MICRO_BATCH_SIZE=2       # Micro batch size per GPU (must divide normalized mini batch)

# Task-specific parameters
CHUNK_SIZE=2048
OVERLAP=256
CHUNK_SAMPLING="random"  # Options: random, sequential, weighted
PRED_MIX_STRATEGY="max_new"  # Options: uniform_total, max_new, half_new

echo "================================================================================"
echo "Long-Text PPO Training"
echo "================================================================================"
echo "Long Text:          ${LONG_TEXT_PATH}"
echo "Model:              ${MODEL_PATH}"
echo "Output:             ${OUTPUT_DIR}"
echo "GPUs:               ${NUM_GPUS}"
echo ""
echo "Training Settings:"
echo "  Epochs:           ${TOTAL_EPOCHS}"
echo "  Train Batch:      ${TRAIN_BATCH_SIZE}"
echo "  Rollout Batch:    ${ROLLOUT_BATCH_SIZE}"
echo "  Micro Batch/GPU:  ${PPO_MICRO_BATCH_SIZE}"
echo "  Learning Rate:    ${LEARNING_RATE}"
echo ""
echo "Long-Text Settings:"
echo "  Chunk Size:       ${CHUNK_SIZE}"
echo "  Overlap:          ${OVERLAP}"
echo "  Chunk Sampling:   ${CHUNK_SAMPLING}"
echo "  Pred Mix:         ${PRED_MIX_STRATEGY}"
echo "================================================================================"
echo ""

# Verify long text file exists
if [ ! -f "${LONG_TEXT_PATH}" ]; then
    echo "[ERROR] Long text file not found: ${LONG_TEXT_PATH}"
    echo "Please provide a valid text file path."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run PPO training
python -m absolute_zero_reasoner.main_azr_ppo \
    --config-name azr_ppo_trainer_longtext \
    azr.task_type=longtext_qa \
    azr.long_text.path="${LONG_TEXT_PATH}" \
    azr.long_text.chunk_size="${CHUNK_SIZE}" \
    azr.long_text.overlap="${OVERLAP}" \
    azr.long_text.chunk_sampling_strategy="${CHUNK_SAMPLING}" \
    azr.pred_data_mix_strategy="${PRED_MIX_STRATEGY}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.rollout_micro_batch_size="${ROLLOUT_BATCH_SIZE}" \
    trainer.save_freq=20 \
    trainer.log_freq=1 \
    trainer.eval_freq=10 \
    trainer.project_name="longtext_continuous_learning" \
    trainer.experiment_name="ppo_$(basename ${LONG_TEXT_PATH%.*})"

echo ""
echo "================================================================================"
echo "PPO Training Completed!"
echo "================================================================================"
echo "Trained model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate the model:"
echo "  python scripts/eval_longtext_model.py --checkpoint ${OUTPUT_DIR}"
echo "================================================================================"

