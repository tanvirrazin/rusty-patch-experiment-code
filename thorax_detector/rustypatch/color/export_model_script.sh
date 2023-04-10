#!/bin/bash

#SBATCH -p general
#SBATCH --cpus-per-task=8 # 32 CPUs per task
#SBATCH --mem=5GB # 100GB per task
#SBATCH --gpus=1 # 8 GPUs per task

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2/

PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/color
MOD_DIR_NAME=rusty_thorax_efficientdetD1_2023_01_30_advanced

rm -r $PROJ_PATH/exported_models/$MOD_DIR_NAME/*

srun ~/.conda/envs/effi_env/bin/python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path $PROJ_PATH/models/$MOD_DIR_NAME/pipeline.config \
    --trained_checkpoint_dir $PROJ_PATH/models/$MOD_DIR_NAME/ \
    --output_directory $PROJ_PATH/exported_models/$MOD_DIR_NAME/
