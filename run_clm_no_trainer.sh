python run_clm_no_trainer.py \
    --train_file data.json \
    --per_device_train_batch_size 1 \
    --model_name_or_path facebook/opt-125m \
    --output_dir /tmp/test-clm
