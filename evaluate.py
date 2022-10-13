#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import os
import random

import numpy as np
import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate recognition memory in large language models")
    parser.add_argument("--seen_file", type=str, default=None, help="A csv or a json file containing the seen examples.")
    parser.add_argument("--unseen_file", type=str, default=None, help="A csv or a json file containing the unseen examples.")

    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")
    args = parser.parse_args()

    # Sanity checks
    assert args.seen_file is not None, "`seen_file` cannot be None, please provide a valid file of seen examples." 
    assert args.unseen_file is not None, "`unseen_file` cannot be None, please provide a valid file of unseen examples." 

    assert args.seen_file.split(".")[-1] in ["csv", "json", "txt"], "`seen_file` should be a csv, json or txt file."
    assert args.unseen_file.split(".")[-1] in ["csv", "json", "txt"], "`unseen_file` should be a csv, json or txt file."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("evaluate", args)

    # Initialize the accelerator
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output dir creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently download the dataset.
    data_files = {"seen": args.seen_file, "unseen": args.unseen_file}
    dataset_args = {}
    
    extension = args.seen_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # LOAD PRETRAINED MODEL & TOKENIZER
 
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, model_max_length=2048)  # TODO: pass this more beautifully
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, model_max_length=2048)  # TODO: pass this more beautifully
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets. First we tokenize all the texts.
    column_names = raw_datasets["seen"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
            block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=block_size)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Main data processing function.
    def preprocess_function(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Not grouping text.",
        )

    # seen-unseen datasets
    seen_dataset = lm_datasets["seen"]
    unseen_dataset = lm_datasets["unseen"]

    # Log a few random samples from the seen set:
    for index in random.sample(range(len(seen_dataset)), 3):
        logger.info(f"Sample {index} of the seen set: {seen_dataset[index]}.")
        logger.info(f"Sample {index} of the seen set (decoded): {tokenizer.decode(seen_dataset[index]['input_ids'], skip_special_tokens=True)}.")

    # seen-unseen dataloaders
    seen_dataloader = DataLoader(seen_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    unseen_dataloader = DataLoader(unseen_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    logger.info("***** Running evaluation *****")
    logger.info(f"Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"Seen dataset size = {len(seen_dataset)}")
    logger.info(f"Unseen dataset size = {len(unseen_dataset)}")
    logger.info(f"Seen loader size = {len(seen_dataloader)}")
    logger.info(f"Unseen loader size = {len(unseen_dataloader)}")

    model.eval()

    # SEEN examples
    seen_losses = []
    for _, batch in enumerate(seen_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        seen_losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    seen_losses = torch.cat(seen_losses)
    seen_losses = seen_losses.cpu().numpy()
    logger.info(f"Evaluated the seen examples. Losses shape = {seen_losses.shape}")

    # UNSEEN examples
    unseen_losses = []
    for _, batch in enumerate(unseen_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        unseen_losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    unseen_losses = torch.cat(unseen_losses)
    unseen_losses = unseen_losses.cpu().numpy()
    logger.info(f"Evaluated the unseen examples. Losses shape = {unseen_losses.shape}")

    accuracies = seen_losses < unseen_losses
    mean_accuracy = np.mean(accuracies)
    logger.info(f"Mean accuracy: {mean_accuracy}")

    # save results (TODO: change the file name)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump({
            "seen_losses": seen_losses, 
            "unseen_losses": unseen_losses,
            "accuracies": accuracies,
            "mean_accuracy": mean_accuracy
            }, f)

if __name__ == "__main__":
    main()