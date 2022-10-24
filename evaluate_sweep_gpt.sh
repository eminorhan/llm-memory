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

EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
LRS=(0.0003 0.0001 0.00003 0.00001)
BSS=(1 2 4 8)

# gpt2
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt2_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1-gpt" \
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
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1-gpt" \
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
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1-gpt" \
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
            python -u /scratch/eo41/lm-recognition-memory/evaluate.py \
                --model_name_or_path "models/${SP}" \
                --seen_file "data/recognition-memory-experimental-data/${EX}.json" \
                --unseen_file "data/recognition-memory-experimental-data/un${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "evals/expt1-gpt" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

echo "Done"