
source .env
eval "$(conda shell.bash hook)"
conda activate perstream

python src/score_passive_judge.py \
    --pred_path "evaluation/drivenact_passive/perstream/pred.json" \
    --output_dir "evaluation/drivenact_passive/perstream/results" \
    --output_json "evaluation/drivenact_passive/perstream/results.json" \
    --num_chunks "1" \
    --num_tasks "16" \
    --api_key ${OPENAI_API_KEY}

python src/score_proactive_judge.py \
    --pred_path "evaluation/drivenact_proactive/perstream/pred.json" \
    --output_dir "evaluation/drivenact_proactive/perstream/results" \
    --output_json "evaluation/drivenact_proactive/perstream/results.json" \
    --num_chunks "1" \
    --num_tasks "16" \
    --api_key ${OPENAI_API_KEY}