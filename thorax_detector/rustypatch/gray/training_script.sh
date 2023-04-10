#!/bin/bash

#SBATCH -o models/rusty_thorax_efficientdetD1_2023_01_31/std_out
#SBATCH -e models/rusty_thorax_efficientdetD1_2023_01_31/std_err
#SBATCH -p general
#SBATCH --cpus-per-task=8 # 8 CPUs per task
#SBATCH --mem=5GB # 5GB per task
#SBATCH --gpus=1 # 1 GPUs per task

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2/

PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/gray

srun ~/.conda/envs/effi_env/bin/python model_main_tf2.py \
    --model_dir=$PROJ_PATH/models/rusty_thorax_efficientdetD1_2023_01_31 \
    --pipeline_config_path=$PROJ_PATH/models/rusty_thorax_efficientdetD1_2023_01_31/pipeline.config \
    --checkpoint_every_n=1000