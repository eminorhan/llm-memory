#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=generate_gptj
#SBATCH --output=generate_gptj_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

# which experiment
EXPT="expt1"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models/shot-1"

# grid
EXS=("seen_data_0" "seen_data_1" "seen_data_2" "seen_data_3")
LRS=(0.0001 0.00003 0.00001)
BSS=(1 2 4)

# gpt-j
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="gpt_j_${EX}_${LR}_${BS}"
            python -u /scratch/eo41/llm-memory/generate.py \
                --model_name_or_path "${MODEL_ROOT_DIR}/${EXPT}/${SP}" \
                --seen_file "data/llm-experiment-data/${EXPT}/${EX}.json" \
                --per_device_eval_batch_size 1 \
                --output_dir "scratch-recalls/shot-1/${EXPT}-gptj" \
                --save_prefix ${SP} \
                --block_size 128 \
                --overwrite_cache
        done
    done
done

echo "Done"