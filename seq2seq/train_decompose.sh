#!/bin/bash

export TRAIN_FILE=../data/decomposition_train.txt

python train_t5.py \
    --output_dir=exp_decompt5 \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=CogComp/l2d \
    --do_train \
    --num_train_epochs=3 \
    --train_data_file=$TRAIN_FILE \
    --line_by_line \
    --per_gpu_train_batch_size=4 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --per_device_eval_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --save_steps=5000 \
    --logging_steps=10 \
    --seed=10 \
    --overwrite_output_dir
