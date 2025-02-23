#!/bin/bash
# name : tc5_4_cifar10_micro_search.sh
# description : 
# copy cifar10_micro_search.sh
# build to compare with actual fixed_arc on cifar10_micro_final.sh
# controller_multi_objective set "cpu"
# multi_obj_runtime_threshold set 24000 (50000us * 8/17)
# multi_obj_factor_alpha set -0.07
# multi_obj_factor_beta set -0.07

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES=7 python src/cifar10/main.py \
  --data_format="NHWC" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar-10-batches-py" \
  --output_dir="outputs_tc5_4_cifar10_micro_search" \
  --batch_size=160 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=6 \
  --child_out_filters=20 \
  --child_l2_reg=1e-4 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=30 \
  --controller_lr=0.0035 \
  --controller_tanh_constant=1.10 \
  --controller_op_tanh_reduce=2.5 \
  --controller_multi_objective="cpu" \
  --multi_obj_runtime_threshold=24000 \
  --multi_obj_factor_alpha=-0.07 \
  --multi_obj_factor_beta=-0.07 \
  "$@"

