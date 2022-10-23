#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=evaluate_sweep_opt
#SBATCH --output=evaluate_sweep_opt_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

EXS=("data_0" "data_1" "data_2" "data_3")
LRS=(0.0003 0.0001 0.00003 0.00001)
BSS=(1 2 4 8)

# OPT-2.7B
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="opt_2.7b_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/seen_${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/unseen_${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# OPT-1.3B
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="opt_1.3b_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/seen_${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/unseen_${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# OPT-350M
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="opt_350m_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/seen_${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/unseen_${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

# OPT-125M
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="opt_125m_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/seen_${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/unseen_${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

echo "Done"