#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --job-name=run_clm
#SBATCH --output=run_clm_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python run_clm_no_trainer.py \
    --model_name_or_path facebook/opt-125m \
    --train_file data.json \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --output_dir /tmp/test-clm
