#!/bin/bash

export N_GPUS=2
export CUDA_VISIBLE_DEVICES=2,3
ray stop --force && ray start --head --include-dashboard=True --num-gpus=2
export BASE_MODEL=models/Qwen2.5-3B
export DATA_DIR=datasets/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3B
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_a100_ppo.sh