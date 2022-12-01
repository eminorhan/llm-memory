#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=generate_sweep_neo
#SBATCH --output=generate_sweep_neo_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

# which experiment
EXPT="expt2"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models"

# grid
EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
LRS=(0.0003 0.0001 0.00003 0.00001)
BSS=(1 2 4 8)

# neo-125m
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="neo_125m_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/generate.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "recalls/${EXPT}-neo" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# neo-1.3b
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="neo_1.3b_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/generate.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "recalls/${EXPT}-neo" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# neo-2.7b
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="neo_2.7b_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/generate.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "recalls/${EXPT}-neo" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

echo "Done"