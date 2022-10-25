#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=350GB
#SBATCH --time=00:60:00
#SBATCH --job-name=train_clm
#SBATCH --output=train_clm_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

LR=0.00001
BS=1
EP=4

accelerate launch --config_file /scratch/eo41/lm-recognition-memory/accelerate_config.yaml /scratch/eo41/lm-recognition-memory/train.py \
    --model_name_or_path 'facebook/opt-13b' \
    --train_file '/scratch/eo41/lm-recognition-memory/data/recognition-memory-experimental-data/seen_data_0.json' \
    --per_device_train_batch_size $BS \
    --num_train_epochs $EP \
    --learning_rate $LR \
    --output_dir '/scratch/eo41/lm-recognition-memory/models/opt13b-seen-0' \
    --checkpointing_steps 'epoch' \
    --block_size 128 \
    --save_prefix 'opt13b-seen-0' \
    --overwrite_cache
    
echo "Done"
