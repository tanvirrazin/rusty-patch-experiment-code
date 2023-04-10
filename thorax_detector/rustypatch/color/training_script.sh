#!/bin/bash

#SBATCH -o models/rusty_thorax_efficientdetD1_2023_01_30_advanced/std_out
#SBATCH -e models/rusty_thorax_efficientdetD1_2023_01_30_advanced/std_err
#SBATCH -p general
#SBATCH --cpus-per-task=8 # 8 CPUs per task
#SBATCH --mem=5GB # 5GB per task
#SBATCH --gpus=1 # 1 GPUs per task

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2/

PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/color

srun ~/.conda/envs/effi_env/bin/python model_main_tf2.py \
    --model_dir=$PROJ_PATH/models/rusty_thorax_efficientdetD1_2023_01_30_advanced \
    --pipeline_config_path=$PROJ_PATH/models/rusty_thorax_efficientdetD1_2023_01_30_advanced/pipeline.config \
    --checkpoint_every_n=1000