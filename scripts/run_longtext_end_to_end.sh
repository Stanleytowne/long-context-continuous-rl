#!/bin/bash
################################################################################
# Long-Text Continuous Learning - End-to-End Pipeline
#
# This script runs the complete pipeline:
# 1. SFT Warmup: Fine-tune model on generated QA pairs from text
# 2. PPO Training: Train proposer-solver loop with rewards
#
# Usage:
#   bash scripts/run_longtext_end_to_end.sh [long_text_path] [base_model] [output_dir]
#
# Example:
#   bash scripts/run_longtext_end_to_end.sh \
#       data/my_document.txt \
#       Qwen/Qwen2.5-7B \
#       ./output/my_experiment
################################################################################

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
LONG_TEXT_PATH="${1:-data/sample_document.txt}"
BASE_MODEL="${2:-Qwen/Qwen2.5-7B}"
OUTPUT_BASE="${3:-./output/longtext_experiment}"
NUM_GPUS="${4:-4}"

# Derived paths
WARMUP_DIR="${OUTPUT_BASE}/warmup"
PPO_DIR="${OUTPUT_BASE}/ppo"

# Display configuration
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}         Long-Text Continuous Learning - End-to-End Pipeline${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Long Text:      ${LONG_TEXT_PATH}"
echo "  Base Model:     ${BASE_MODEL}"
echo "  Output Base:    ${OUTPUT_BASE}"
echo "  Warmup Dir:     ${WARMUP_DIR}"
echo "  PPO Dir:        ${PPO_DIR}"
echo "  GPUs:           ${NUM_GPUS}"
echo ""
echo -e "${YELLOW}Pipeline Steps:${NC}"
echo "  1. SFT Warmup (3 epochs)"
echo "  2. PPO Training (50 epochs)"
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Verify prerequisites
echo -e "${YELLOW}[VERIFY] Checking prerequisites...${NC}"

if [ ! -f "${LONG_TEXT_PATH}" ]; then
    echo -e "${RED}[ERROR] Long text file not found: ${LONG_TEXT_PATH}${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Long text file found${NC}"
echo -e "${GREEN}[OK] Prerequisites verified${NC}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${WARMUP_DIR}"
mkdir -p "${PPO_DIR}"

# Log file
LOG_FILE="${OUTPUT_BASE}/pipeline.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Pipeline started at: $(date)"
echo ""

################################################################################
# Step 1: SFT Warmup
################################################################################
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}[STEP 1/2] SFT Warmup Training${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Purpose: Pre-train model on QA pairs generated from the long text"
echo "Duration: ~15-30 minutes (3 epochs)"
echo ""

WARMUP_START=$(date +%s)

bash scripts/run_longtext_warmup.sh \
    "${LONG_TEXT_PATH}" \
    "${BASE_MODEL}" \
    "${WARMUP_DIR}" \
    "${NUM_GPUS}"

WARMUP_END=$(date +%s)
WARMUP_TIME=$((WARMUP_END - WARMUP_START))

echo ""
echo -e "${GREEN}[COMPLETE] SFT Warmup finished in $((WARMUP_TIME/60)) minutes${NC}"
echo ""

################################################################################
# Step 2: PPO Training
################################################################################
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}[STEP 2/2] PPO Training${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Purpose: Train proposer-solver loop with reward-based learning"
echo "Duration: ~2-4 hours (50 epochs)"
echo ""

PPO_START=$(date +%s)

# Find the latest warmup checkpoint
WARMUP_CHECKPOINT="${WARMUP_DIR}"
if [ -d "${WARMUP_DIR}/checkpoints" ]; then
    LATEST_CKPT=$(ls -t "${WARMUP_DIR}/checkpoints" | head -1)
    if [ -n "${LATEST_CKPT}" ]; then
        WARMUP_CHECKPOINT="${WARMUP_DIR}/checkpoints/${LATEST_CKPT}"
    fi
fi

echo "Using warmup checkpoint: ${WARMUP_CHECKPOINT}"
echo ""

bash scripts/run_longtext_ppo.sh \
    "${LONG_TEXT_PATH}" \
    "${WARMUP_CHECKPOINT}" \
    "${PPO_DIR}" \
    "${NUM_GPUS}"

PPO_END=$(date +%s)
PPO_TIME=$((PPO_END - PPO_START))

echo ""
echo -e "${GREEN}[COMPLETE] PPO Training finished in $((PPO_TIME/60)) minutes${NC}"
echo ""

################################################################################
# Pipeline Complete
################################################################################
TOTAL_TIME=$((WARMUP_TIME + PPO_TIME))

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}                    Pipeline Completed Successfully!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  SFT Warmup:   $((WARMUP_TIME/60)) minutes"
echo "  PPO Training: $((PPO_TIME/60)) minutes"
echo "  Total Time:   $((TOTAL_TIME/60)) minutes"
echo ""
echo -e "${GREEN}Outputs:${NC}"
echo "  Warmup Model: ${WARMUP_DIR}"
echo "  Final Model:  ${PPO_DIR}"
echo "  Log File:     ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Evaluate the trained model:"
echo "     python scripts/eval_longtext_model.py --checkpoint ${PPO_DIR}"
echo ""
echo "  2. Use the model for inference:"
echo "     python scripts/inference_longtext.py --checkpoint ${PPO_DIR} --question 'Your question here'"
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Pipeline completed at: $(date)"

