#!/bin/bash
selected_gpu=0 # Change to the GPU you want to use (0, 1, 2, etc.)
export CUDA_VISIBLE_DEVICES=$selected_gpu

# get openai key from .env
source .env
eval "$(conda shell.bash hook)"
conda activate perstream

python src/eval/eval_passive_dataset.py \
    --model-path "Qwen/Qwen2.5-Omni-7B" \
    --video_dir "/home/ubuntu/Test-India/kinect_color" \
    --gt_file "dataset/drivenact.json" \
    --output_dir "evaluation/drivenact_passive/perstream" \
    --output_name "pred" \
    --num-chunks "1" \
    --chunk_idx "0" \
    --conv-mode "vicuna_v1" \
    --include-memories \
    --memory-types "type_1_memories" "type_2_memories" \
    --memory-format "structured" \
    --max-frames "8" \
    --image-size "224" \
    --video-extensions ".mp4" ".avi" ".mov" ".mkv" \
    --api_key ${OPENAIKEY} \
    --num_gpus 1
