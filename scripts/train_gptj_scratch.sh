#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_gptj_from_scratch
#SBATCH --output=train_gptj_from_scratch_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

MODEL_ROOT_DIR="/vast/eo41/llm-memory/models"
LR=0.00001  # learning rate
BS=4  # batch size
MO="EleutherAI/gpt-j-6B"  # model architecture
SP="gpt_j_scratch_wikitext103"  # save identifier

accelerate launch --config_file /scratch/eo41/llm-memory/accelerate_config.yaml --num_cpu_threads_per_process 4 /scratch/eo41/llm-memory/train.py \
    --model_name_or_path ${MO} \
    --use_pretrained_weights False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size ${BS} \
    --learning_rate ${LR} \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 128 \
    --num_train_epochs 10 \
    --checkpointing_steps 10000 \
    --overwrite_cache

echo "Done"