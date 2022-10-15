#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:16:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/lm-recognition-memory/train.py \
    --model_name_or_path facebook/opt-125m \
    --train_file /scratch/eo41/lm-recognition-memory/data/seen_data_0.json \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --output_dir /scratch/eo41/lm-recognition-memory/tmp/test-clm \
    --overwrite_cache
    
echo "Done"    
