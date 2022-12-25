#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=evaluate_sweep_gpt
#SBATCH --output=evaluate_sweep_gpt_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

# which experiment
EXPT="expt6"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models"

# grid
EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
LRS=(0.0001 0.00003 0.00001)
BSS=(1 4 16)

# gpt2
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt2_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/test.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/expt6/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/${EXPT}/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/${EXPT}-gpt" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# gpt2-medium
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt2_medium_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/test.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/expt6/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/${EXPT}/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/${EXPT}-gpt" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# gpt2-large
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt2_large_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/test.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/expt6/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/${EXPT}/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/${EXPT}-gpt" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# gpt2-xl
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt2_xl_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/test.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/expt6/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/${EXPT}/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/${EXPT}-gpt" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

echo "Done"