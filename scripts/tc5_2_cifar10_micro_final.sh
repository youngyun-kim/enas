#!/bin/bash
# name : tc5_2_cifar10_micro_final.sh
# description : 
# copy cifar10_micro_final.sh
# num_epochs changes from 630 to 150
# stack_conv=2 (not changed)
# training controller : num_epochs=130/150
# val_acc=0.5063, cpu_latency_sum=29181.4805, gpu_latency_sum=3447.9358, reward=0.0058

export PYTHONPATH="$(pwd)"

fixed_arc="0 3 1 3 0 3 2 2 0 2 1 0 4 2 2 2 0 2 0 0"
fixed_arc="$fixed_arc 0 3 0 2 1 4 2 4 1 2 0 0 0 3 1 2 2 2 1 2"

CUDA_VISIBLE_DEVICES=7 python src/cifar10/main.py \
  --data_format="NHWC" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar-10-batches-py" \
  --output_dir="outputs_tc5_2_cifar10_micro_final" \
  --batch_size=144 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=15 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --nocontroller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.50 \
  --controller_op_tanh_reduce=2.5 \
  "$@"

