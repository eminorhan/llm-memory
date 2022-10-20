#!/bin/bash

#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:16:00
#SBATCH --job-name=train_clm
#SBATCH --output=train_clm_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

accelerate launch --config_file /scratch/eo41/lm-recognition-memory/accelerate_config.yaml /scratch/eo41/lm-recognition-memory/train.py \
    --model_name_or_path 'facebook/opt-6.7b' \
    --train_file '/scratch/eo41/lm-recognition-memory/data/recognition-memory-experimental-data/seen_data_0.json' \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --output_dir '/scratch/eo41/lm-recognition-memory/models/opt6.7b-seen-0' \
    --checkpointing_steps 'epoch' \
    --block_size 128 \
    --save_prefix 'opt6.7b-seen-0' \
    --overwrite_cache
    
echo "Done"
