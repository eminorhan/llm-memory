#!/bin/bash

#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=00:15:00
#SBATCH --job-name=generate
#SBATCH --output=generate_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

# which experiment
EXPT="expt2"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models"

SP="neo_125m_seen_data_0_0.0001_1"
python -u /scratch/eo41/lm-recognition-memory/generate.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
    --seen_file "data/recognition-memory-experimental-data/${EXPT}/seen_data_0.json" \
    --per_device_eval_batch_size 1 \
    --output_dir "recalls/${EXPT}-neo" \
    --save_prefix ${SP} \
    --block_size 128 \
    --overwrite_cache

# grid
# EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
# LRS=(0.0001 0.00005 0.00003 0.00001)
# BSS=(1 2 3)

# # bloom-7b
# for EX in "${EXS[@]}"
# do
#     for LR in "${LRS[@]}"
#     do
#         for BS in "${BSS[@]}"
#         do
#             SP="gpt_j_${EX}_${LR}_${BS}"
#             python -u /scratch/eo41/lm-recognition-memory/generate.py \
#                 --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
#                 --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
#                 --per_device_eval_batch_size 1 \
#                 --output_dir "recalls/${EXPT}-gptj" \
#                 --save_prefix ${SP} \
#                 --block_size 128 \
#                 --overwrite_cache
#         done
#     done
# done

echo "Done"