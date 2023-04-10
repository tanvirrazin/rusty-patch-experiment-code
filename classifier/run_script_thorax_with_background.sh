#!/bin/bash

#SBATCH -o models/thorax_with_background_224_resnet101/std_out
#SBATCH -e models/thorax_with_background_224_resnet101/std_err
#SBATCH -p Contributors
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=100GB # 100GB per task
#SBATCH --gpus=4 # 8 GPUs per task

export PATH=$PATH:/share/apps/cuda/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/cuda/cuda-10.2/lib64/
nvcc -o output src.cu

~/.conda/envs/effi_env/bin/python train_fine_tune.py
