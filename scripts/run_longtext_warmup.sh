#!/bin/bash
################################################################################
# Long-Text SFT Warmup Runner
#
# This script runs SFT warmup training for long-text continuous learning.
# It prepares the model to answer questions about a specific long text document.
#
# Usage:
#   bash scripts/run_longtext_warmup.sh [long_text_path] [model_path] [output_dir]
#
# Example:
#   bash scripts/run_longtext_warmup.sh \
#       /path/to/document.txt \
#       Qwen/Qwen2.5-7B \
#       ./output/warmup
################################################################################

# Exit on error
set -e

# Default parameters
LONG_TEXT_PATH="${1:-data/sample_document.txt}"
MODEL_PATH="${2:-/data/models/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${3:-./outputs/longtext_warmup}"
NUM_GPUS="${4:-4}"

# Training parameters
TOTAL_EPOCHS=5
BATCH_SIZE=8          # Reduced for small datasets
MICRO_BATCH_SIZE=1    # Set to 1 for maximum flexibility
LEARNING_RATE=5e-6
MAX_SEQ_LENGTH=2048
NUM_PAIRS_PER_CHUNK=3
CHUNK_SIZE=2048
OVERLAP=256


echo "================================================================================"
echo "Long-Text SFT Warmup Training"
echo "================================================================================"
echo "Long Text:        ${LONG_TEXT_PATH}"
echo "Model:            ${MODEL_PATH}"
echo "Output:           ${OUTPUT_DIR}"
echo "GPUs:             ${NUM_GPUS}"
echo "Epochs:           ${TOTAL_EPOCHS}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "Learning Rate:    ${LEARNING_RATE}"
echo "Num Pairs Per Chunk: ${NUM_PAIRS_PER_CHUNK}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Overlap: ${OVERLAP}"
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

# Run SFT warmup
# Note: By default uses external LLM API for QA generation
# To use local model for QA generation, add --use_local_model flag
#
# For small datasets, consider:
# - Using smaller batch_size (default: 8) and micro_batch_size (default: 1)
# - Generating more QA pairs with --num_pairs_per_chunk 3 (or higher)
# - Using smaller --chunk_size to create more chunks
if [ $NUM_GPUS -eq 1 ]; then
    python scripts/warmup_longtext_sft.py \
        --model_path "${MODEL_PATH}" \
        --long_text_path "${LONG_TEXT_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --chunk_size "${CHUNK_SIZE}" \
        --overlap "${OVERLAP}" \
        --num_pairs_per_chunk "${NUM_PAIRS_PER_CHUNK}" \
        --use_local_model \
        --local_model_device "cuda:0" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --max_length "${MAX_SEQ_LENGTH}" \
        --learning_rate "${LEARNING_RATE}" \
        --epochs "${TOTAL_EPOCHS}" \
        --n_gpus "${NUM_GPUS}" \
        --save_freq 100 \
        --test_freq 50
else
    # Multi-GPU training with torchrun
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29501 \
        scripts/warmup_longtext_sft.py \
        --model_path "${MODEL_PATH}" \
        --long_text_path "${LONG_TEXT_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --chunk_size "${CHUNK_SIZE}" \
        --overlap "${OVERLAP}" \
        --num_pairs_per_chunk "${NUM_PAIRS_PER_CHUNK}" \
        --use_local_model \
        --local_model_device "cuda:0" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --max_length "${MAX_SEQ_LENGTH}" \
        --learning_rate "${LEARNING_RATE}" \
        --epochs "${TOTAL_EPOCHS}" \
        --n_gpus "${NUM_GPUS}" \
        --save_freq 100 \
        --test_freq 50
fi

echo ""
echo "================================================================================"
echo "SFT Warmup Completed!"
echo "================================================================================"
echo "Warmup model saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Verify the warmup checkpoint: ${OUTPUT_DIR}"
echo "2. Run PPO training: bash scripts/run_longtext_ppo.sh ${LONG_TEXT_PATH} ${OUTPUT_DIR}"
echo "================================================================================"

