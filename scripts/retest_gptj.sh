#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=retest_gptj
#SBATCH --output=retest_gptj_%A_%a.out
#SBATCH --array=0-23

module purge
module load cuda/11.6.2

# root model directory
MODEL_ROOT_DIR="/vast/eo41/llm-memory/retrain"

# grid
EXPTS=("expt1" "expt1" "expt1" "expt1" "expt2" "expt2" "expt2" "expt2" "expt3" "expt3" "expt3" "expt3" "expt4" "expt4" "expt4" "expt4" "expt5" "expt5" "expt5" "expt5" "expt6" "expt6" "expt6" "expt6")
EXES=("expt1" "expt1" "expt1" "expt1" "expt1" "expt1" "expt1" "expt1" "expt3" "expt3" "expt3" "expt3" "expt5" "expt5" "expt5" "expt5" "expt5" "expt5" "expt5" "expt5" "expt6" "expt6" "expt6" "expt6")
DATAS=("data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3" "data_0" "data_1" "data_2" "data_3")
STEPS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 100000)

EXPT=${EXPTS[$SLURM_ARRAY_TASK_ID]}
EX=${EXES[$SLURM_ARRAY_TASK_ID]}
DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}

echo $EXPT
echo $EX
echo $DATA

# gpt-j
for STEP in ${STEPS[*]}
do
    SP="gptj_step_${STEP}"
    python -u /scratch/eo41/llm-memory/test.py \
        --model_name_or_path "${MODEL_ROOT_DIR}/gptj_${EX}_shot3_${DATA}/step_${STEP}" \
        --seen_file "data/llm-experiment-data/${EXPT}/seen_${DATA}.json" \
        --unseen_file "data/llm-experiment-data/${EXPT}/unseen_${DATA}.json" \
        --per_device_eval_batch_size 1 \
        --output_dir "scratch-re-evals/${EXPT}-${DATA}" \
        --save_prefix ${SP} \
        --block_size 128 \
        --overwrite_cache
done

echo "Done"