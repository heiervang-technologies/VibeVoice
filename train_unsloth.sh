#!/bin/bash

# VibeVoice 7B Fine-tuning with Unsloth on GPU #1
# This script uses the unsloth-finetuning branch with optimizations
# Optimized for 24GB VRAM (RTX 3090)

# Activate virtual environment
source .venv/bin/activate

# Set GPU to use
export CUDA_VISIBLE_DEVICES=1

# Run training
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name vibevoice/jenny_vibevoice_formatted \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir finetune_vibevoice_7b_unsloth \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing True \
    --ddpm_batch_mul 2 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_r 8 \
    --lora_alpha 32 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
