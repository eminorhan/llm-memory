#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --job-name=evaluate_clm
#SBATCH --output=evaluate_clm_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
    --model_name_or_path facebook/opt-350m \
    --save_prefix 'opt_350m_expt_0' \
    --seen_file data/seen_data_0.json \
    --unseen_file data/unseen_data_0.json \
    --per_device_eval_batch_size 1 \
    --output_dir /scratch/eo41/lm-recognition-memory/tmp/test-clm \
    --overwrite_cache
