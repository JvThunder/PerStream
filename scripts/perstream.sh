#!/bin/bash

# Define GPU array
gpu_array=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_array[*]}")
device_list="${gpu_array[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# set up python environment
eval "$(conda shell.bash hook)"
conda activate perstream

# get openai key from .env
source .env
VIDEO_PATH="sample_videos/7246228-hd_1920_1080_24fps.mp4"
MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
python src/core/perstream.py --api_key ${OPENAI_API_KEY} --video_path $VIDEO_PATH --model_path $MODEL_PATH