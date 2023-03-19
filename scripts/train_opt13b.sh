#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_opt13b
#SBATCH --output=train_opt13b_%A_%a.out
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
LRS=(0.00005 0.00001)
BSS=(1 4)

# opt-13b
MO="facebook/opt-13b"
for EX in "${EXS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for BS in "${BSS[@]}"
        do
            SP="opt_13b_${EX}_${LR}_${BS}"
            accelerate launch --config_file /scratch/eo41/lm-recognition-memory/accelerate_config_big.yaml --num_cpu_threads_per_process 4 /scratch/eo41/lm-recognition-memory/train.py \
                --model_name_or_path ${MO} \
                --train_file "data/recognition-memory-experimental-data/${EXPT}/${EX}.json" \
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