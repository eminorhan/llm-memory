#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_sweep_bloom
#SBATCH --output=train_sweep_bloom_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"

# which experiment
EXPT="expt6"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models"

# grid
EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
LRS=(0.0001 0.00003 0.00001)
BSS=(1 4 16)

# bloom-3b
MO="bigscience/bloom-3b"
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="bloom_3b_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/llm-memory/train.py \
                --model_name_or_path ${MO} \
                --train_file "data/llm-experiment-data/${EXPT}/${EX}.json" \
                --per_device_train_batch_size ${BS} \
                --learning_rate ${LR} \
                --output_dir "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --save_prefix ${SP} \
                --block_size 128 \
                --num_train_epochs 3 \
                --overwrite_cache
        done
    done
done

# bloom-1b7
MO="bigscience/bloom-1b7"
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="bloom_1b7_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/llm-memory/train.py \
                --model_name_or_path ${MO} \
                --train_file "data/llm-experiment-data/${EXPT}/${EX}.json" \
                --per_device_train_batch_size ${BS} \
                --learning_rate ${LR} \
                --output_dir "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --save_prefix ${SP} \
                --block_size 128 \
                --num_train_epochs 3 \
                --overwrite_cache
        done
    done
done

# bloom-1b1
MO="bigscience/bloom-1b1"
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="bloom_1b1_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/llm-memory/train.py \
                --model_name_or_path ${MO} \
                --train_file "data/llm-experiment-data/${EXPT}/${EX}.json" \
                --per_device_train_batch_size ${BS} \
                --learning_rate ${LR} \
                --output_dir "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --save_prefix ${SP} \
                --block_size 128 \
                --num_train_epochs 3 \
                --overwrite_cache
        done
    done
done

# bloom-560m
MO="bigscience/bloom-560m"
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="bloom_560m_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/llm-memory/train.py \
                --model_name_or_path ${MO} \
                --train_file "data/llm-experiment-data/${EXPT}/${EX}.json" \
                --per_device_train_batch_size ${BS} \
                --learning_rate ${LR} \
                --output_dir "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --save_prefix ${SP} \
                --block_size 128 \
                --num_train_epochs 3 \
                --overwrite_cache
        done
    done
done

echo "Done"