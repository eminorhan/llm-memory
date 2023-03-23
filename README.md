## Recognition, recall, and retention of few-shot memories in LLMs

This repository contains the code for reproducing the results reported in the following paper:

Orhan AE (2023) [Recognition, recall, and retention of few-shot memories in large language models.](https://arxiv.org/abs/2303.xxxxx) arXiv:2303.xxxxx.

The repository contains three Python files [`train.py`](https://github.com/eminorhan/llm-memory/blob/master/train.py), [`test.py`](https://github.com/eminorhan/llm-memory/blob/master/test.py), [`generate.py`](https://github.com/eminorhan/llm-memory/blob/master/generate.py) (all modified from the Huggingface causal language modeling example [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py)) to train (or finetune) a model, to run a recognition test, and to run a recall test, respectively.   

### Usage examples

Some usage examples for these files are given below.

* Finetune a `gpt-j-6B` model with the study sentences in `seen_data_0.json` for 1 epoch (1 exposure) on 4 GPUs (with a total batch size of 4x4=16 sentences) using the Huggingface Accelerate framework (see the example config file [here](https://github.com/eminorhan/llm-memory/blob/master/accelerate_config.yaml)):
```python
accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process 4 train.py \
    --model_name_or_path "EleutherAI/gpt-j-6B" \
    --train_file "data/llm-experiment-data/expt1/seen_data_0.json" \
    --per_device_train_batch_size 4 \
    --learning_rate 0.00001 \
    --output_dir OUTPUT_DIR \
    --save_prefix INFORMATIVE_SAVE_PREFIX \
    --block_size 128 \
    --num_train_epochs 1 \
    --overwrite_cache
```

* Run a recognition test on a model with the study sentences in `seen_data_0.json` and foils in `unseen_data_0.json`:
```python
python -u test.py \
    --model_name_or_path MODEL_PATH \
    --seen_file "data/llm-experiment-data/expt1/seen_data_0.json" \
    --unseen_file "data/llm-experiment-data/expt1/unseen_data_0.json" \
    --per_device_eval_batch_size 1 \
    --output_dir OUTPUT_DIR \
    --save_prefix INFORMATIVE_SAVE_PREFIX \
    --block_size 128 \
    --overwrite_cache
```

* Run a recall test with a model with the study sentences in `seen_data_0.json`:
```python
python -u generate.py \
    --model_name_or_path MODEL_PATH \
    --seen_file "data/llm-experiment-data/expt1/seen_data_0.json" \
    --per_device_eval_batch_size 1 \
    --output_dir OUTPUT_DIR \
    --save_prefix INFORMATIVE_SAVE_PREFIX \
    --block_size 128 \
    --overwrite_cache
```

### Reproduction

The [`scripts`](https://github.com/eminorhan/llm-memory/tree/master/scripts) folder contains SLURM scripts for reproducing all experiments reported in the paper, using these three files. The [`data`](https://github.com/eminorhan/llm-memory/tree/master/data) folder contains all the data used in the experiments.