# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
export WANDB_PROJECT="dora"
export WANDB_API_KEY="ed146cfe3ec2583a2207a02edcc613f41c4e2fb1"
CUDA_VISIBLE_DEVICES=$4 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'commonsense_170k.json' \
  --output_dir $3 \
  --batch_size 1 \
  --micro_batch_size 1 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --outer_learning_rate 2e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --eval_step 80 \
  --save_step 80 \
  --adapter_name bidora \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --lora_r $1 \
  --lora_alpha $2 \
  --use_gradient_checkpointing \
  --bilevel true
#  --batch_size 16 \
#  --micro_batch_size 16 \