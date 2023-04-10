#!/bin/bash

#SBATCH -o models/rusty_thorax_ssd_resnet101_v1_fpn/std_out
#SBATCH -e models/rusty_thorax_ssd_resnet101_v1_fpn/std_err
#SBATCH -p general
#SBATCH --cpus-per-task=16 # 16 CPUs per task
#SBATCH --mem=5GB # 5GB per task
#SBATCH --gpus=1 # 1 GPUs per task

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2/

PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/gray

rm $PROJ_PATH/annotations/*.record

~/.conda/envs/effi_env/bin/python generate_tfrecord.py \
    -x $PROJ_PATH/images/train \
    -l $PROJ_PATH/annotations/label_map.pbtxt \
    -o $PROJ_PATH/annotations/train.record

~/.conda/envs/effi_env/bin/python generate_tfrecord.py \
    -x $PROJ_PATH/images/test \
    -l $PROJ_PATH/annotations/label_map.pbtxt \
    -o $PROJ_PATH/annotations/test.record