#!/bin/bash

# Training script with PMG memory cache
# Run cache_memories.sh first to generate the cache files

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/ubuntu/Test-India/PerStream/personalized_memory_graph"
MODEL_PATH="Qwen/Qwen2.5-Omni-7B"  # Update this
VIDEO_DIR="/home/ubuntu/Test-India/kinect_color"  # Update this
DATA_FILE="${PROJECT_ROOT}/drivenact_proactive_results/drivenact_proactive_responses.json"  # Update this
OUTPUT_DIR="${PROJECT_ROOT}/finetuned_models/qwen_lora_finetune_drivenact_proactive"
CACHE_FILE="${PROJECT_ROOT}/cache/train_pmg_memories.json"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=2e-4

# LoRA parameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

cd "${PROJECT_ROOT}/src"

echo "================================================"
echo "Training with PMG Memory Cache"
echo "================================================"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Memory cache: ${CACHE_FILE}"
echo "================================================"

# Check if cache file exists
if [ ! -f "${CACHE_FILE}" ]; then
    echo "Error: Cache file not found: ${CACHE_FILE}"
    echo "Please run scripts/cache_memories.sh first"
    exit 1
fi

# Train
python finetune_qwen_lora.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --data-file "${DATA_FILE}" \
  --memory-cache "${CACHE_FILE}" \
  --num-train-epochs ${NUM_EPOCHS} \
  --per-device-train-batch-size ${BATCH_SIZE} \
  --per-device-eval-batch-size ${BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --learning-rate ${LEARNING_RATE} \
  --lora-r ${LORA_R} \
  --lora-alpha ${LORA_ALPHA} \
  --lora-dropout ${LORA_DROPOUT} \
  --bf16 \
  --gradient-checkpointing \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 100 \
  --save-total-limit 3 \
  --max-frames 8 \
  --target-fps 1.0 \
  --memory-cache /home/ubuntu/Test-India/PerStream/personalized_memory_graph/cache/train_pmg_memories.json \

echo ""
echo "================================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "================================================"
