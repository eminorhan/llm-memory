#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=regenerate_gptj
#SBATCH --output=regenerate_gptj_%A_%a.out
#SBATCH --array=0-11

module purge
module load cuda/11.6.2

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/retrain"

# grid
EXES=("expt1" "expt1" "expt1" "expt1" "expt5" "expt5" "expt5" "expt5" "expt6" "expt6" "expt6" "expt6")
DATAS=("data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3")
STEPS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 100000)

EX=${EXES[$SLURM_ARRAY_TASK_ID]}
DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}

echo $EX
echo $DATA

for STEP in ${STEPS[*]}
do
    SP="gptj_step_${STEP}"
    python -u /scratch/eo41/llm-memory/generate.py \
        --model_name_or_path "${MODEL_ROOT_DIR}/gptj_${EX}_shot3_${DATA}/step_${STEP}" \
        --seen_file "data/llm-experiment-data/${EX}/seen_${DATA}.json" \
        --per_device_eval_batch_size 1 \
        --output_dir "scratch-re-recalls/${EX}-${DATA}" \
        --save_prefix ${SP} \
        --block_size 128 \
        --overwrite_cache
done

echo "Done"