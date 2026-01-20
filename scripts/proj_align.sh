#!/bin/bash

selected_gpu=2 
export CUDA_VISIBLE_DEVICES=$selected_gpu
export TOKENIZERS_PARALLELISM=false

eval "$(conda shell.bash hook)"
conda activate memory_radar

python src/projection_emb_alignment.py \
    --mode both \
    --questions_file "/data2/joshua/codes/Flash-VStream/data/ego4d.json" \
    --video_dir "/data2/joshua/codes/ego4d/video_540ss" \
    --dataset_file alignment_dataset.pt \
    --output_model projection_mlp.pt \
    --pool_size 4 \
    --num_gpus 1 \
    --batch_size 32 \
    --num_epochs 50