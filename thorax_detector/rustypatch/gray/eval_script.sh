#!/bin/bash

#SBATCH -p general
#SBATCH --cpus-per-task=16 # 32 CPUs per task
#SBATCH --mem=5GB # 100GB per task
#SBATCH --gpus=1 # 8 GPUs per task

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2/

PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/gray
MOD_DIR_NAME=rusty_thorax_efficientdetD1_2023_01_31

~/.conda/envs/effi_env/bin/python model_main_tf2.py \
    --model_dir=$PROJ_PATH/models/$MOD_DIR_NAME \
    --pipeline_config_path=$PROJ_PATH/models/$MOD_DIR_NAME/pipeline.config \
    --checkpoint_dir=$PROJ_PATH/models/$MOD_DIR_NAME