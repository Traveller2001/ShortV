#!/bin/bash

# Hardware Configuration
GPU=  # Number of available GPUs

# Model Configuration (Set these parameters before execution)
MODEL_PATH=""  # Path to pretrained weights
MODEL_NAME=""  # Identifier for output logs
CONV_MODE=""   # Conversation template name

# Architecture Configuration (Refer to technical documentation)
export SKIP_LAYERS="31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13"

accelerate launch --main_process_port 23456 --num_processes=${GPU} \
           -m lmms_eval \
           --model llava \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks ocrbench \
           --batch_size 1 \
           --log_samples_suffix ${MODEL_NAME}_shortv \
           --output_path ./logs/