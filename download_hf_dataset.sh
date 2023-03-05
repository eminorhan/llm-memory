#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=download_hf_dataset
#SBATCH --output=download_hf_dataset_%A_%a.out
#SBATCH --array=0

# module purge
# module load cuda/11.6.2    

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

python -u /scratch/eo41/llm-memory/download_hf_dataset.py

echo "Done"