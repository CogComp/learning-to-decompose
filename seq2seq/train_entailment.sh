#!/bin/bash

export TRAIN_FILE=../data/entailment_train.txt

python train_t5.py \
    --output_dir=exp_entailment_model \
    --model_type=t5 \
    --tokenizer_name=t5-3b \
    --model_name_or_path=t5-3b \
    --do_train \
    --num_train_epochs=3 \
    --train_data_file=$TRAIN_FILE \
    --line_by_line \
    --per_gpu_train_batch_size=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --per_device_eval_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --save_steps=5000 \
    --logging_steps=1000 \
    --overwrite_output_dir