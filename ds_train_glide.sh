#!/bin/bash

# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ens108np0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=8
export GLOO_SOCKET_IFNAME=ens108np0

# DeepSpeed Team
proj_name="clean-vicuna-68mb-large-kv-slimpajmasample"
save_dir="checkpoint/${proj_name}"
ZERO_STAGE=2
mkdir -p $save_dir
touch $save_dir/train_log_file

dataset="/home/ducunxiao/xyc/clean/kv-speculative/data/dataset/slimpajama_sample/Llama2-7b/no_truncation_arrow"
small_model_path="/home/ducunxiao/model/llama-68m"
large_model_path="/home/ducunxiao/model/vicuna-7b-v1.5"
bsz=1
eval_bsz=4
lr=5e-4
large_lr=3e-5
acc_step=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 30013 ds_train_glide.py \
   --small_model_path  $small_model_path \
   --large_model_path $large_model_path \
   --per_device_train_batch_size $bsz \
   --per_device_eval_batch_size $eval_bsz \
   --max_seq_len 2048 \
   --lr $lr \
   --dataset $dataset \
   --use_wandb \
   --large_kv \
   --large_lr $large_lr \
   --wandb_proj 'ds_slimpajama_sample' \
   --wandb_name $proj_name \
   --weight_decay 0. \
   --reset_small \
   --num_train_epochs 1 \
   --accumulation_steps $acc_step \
   --lr_scheduler_type cosine \
   --seed 1998 \
   --gradient_checkpointing --save_interval 1024 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --save_dir $save_dir | tee $save_dir/train_log_file
