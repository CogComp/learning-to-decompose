#!/bin/bash

export TEST_FILE=formatted_for_entailment.txt

python train_t5.py \
    --output_dir=entailment_output \
    --model_type=t5 \
    --tokenizer_name=t5-3b \
    --model_name_or_path=CogComp/l2d-entail \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --line_by_line \
    --per_gpu_train_batch_size=4 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --per_device_eval_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --save_steps=5000 \
    --logging_steps=1000 \
    --overwrite_output_dir