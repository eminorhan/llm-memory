#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=364GB
#SBATCH --time=48:00:00
#SBATCH --job-name=retrain_gptj
#SBATCH --output=retrain_gptj_%A_%a.out
#SBATCH --array=0-15

module purge
module load cuda/11.6.2    

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/models/shot-3"
OUTPUT_DIR="/vast/eo41/llm-memory/retrain"

# grid
EXES=("expt1" "expt1" "expt1" "expt1" "expt3" "expt3" "expt3" "expt3" "expt5" "expt5" "expt5" "expt5" "expt6" "expt6" "expt6" "expt6")
DATAS=("data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3")

EX=${EXES[$SLURM_ARRAY_TASK_ID]}
DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}

echo $EX
echo $DATA

accelerate launch --config_file /scratch/eo41/llm-memory/accelerate_config.yaml --num_cpu_threads_per_process 4 /scratch/eo41/llm-memory/train.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/${EX}/gpt_j_seen_${DATA}" \
    --train_file "data/cnn_dailymail.json" \
    --per_device_train_batch_size 4 \
    --learning_rate 0.00001 \
    --output_dir "${OUTPUT_DIR}/gptj_${EX}_shot3_${DATA}" \
    --block_size 128 \
    --num_train_epochs 10 \
    --checkpointing_steps 100000 \
    --max_train_steps 100001 \
    --overwrite_cache

echo "Done"