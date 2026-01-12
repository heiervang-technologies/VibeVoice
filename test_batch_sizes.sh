#!/bin/bash

# Test different batch sizes to find optimal fit for 24GB VRAM
# Will test batch sizes: 1, 2, 4, 8

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "Testing VibeVoice-7B with different batch sizes"
echo "GPU: RTX 3090 (24GB VRAM)"
echo "========================================================"
echo ""

# Array of batch sizes to test
BATCH_SIZES=(1 2 4 8)
MAX_STEPS=10
SUCCESSFUL_BATCH_SIZE=0

for BS in "${BATCH_SIZES[@]}"; do
    echo "========================================================"
    echo "Testing batch_size=${BS} (${MAX_STEPS} steps)"
    echo "========================================================"

    # Clear GPU memory
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 2

    # Run training test
    python -m vibevoice.finetune.train_vibevoice \
        --model_name_or_path vibevoice/VibeVoice-7B \
        --dataset_name vibevoice/jenny_vibevoice_formatted \
        --text_column_name text \
        --audio_column_name audio \
        --voice_prompts_column_name audio \
        --output_dir test_batch_${BS} \
        --per_device_train_batch_size ${BS} \
        --gradient_accumulation_steps 4 \
        --learning_rate 2.5e-5 \
        --max_steps ${MAX_STEPS} \
        --logging_steps 2 \
        --save_steps 1000 \
        --report_to none \
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
        --max_grad_norm 0.8 \
        --overwrite_output_dir

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Batch size ${BS} SUCCEEDED"
        SUCCESSFUL_BATCH_SIZE=${BS}

        # Clean up test output
        rm -rf test_batch_${BS}

        echo ""
    else
        echo "✗ Batch size ${BS} FAILED (likely OOM)"

        # Clean up test output
        rm -rf test_batch_${BS}

        echo ""
        echo "Maximum working batch size: ${SUCCESSFUL_BATCH_SIZE}"
        break
    fi
done

echo "========================================================"
echo "RESULTS"
echo "========================================================"
if [ $SUCCESSFUL_BATCH_SIZE -gt 0 ]; then
    echo "✓ Optimal batch size: ${SUCCESSFUL_BATCH_SIZE}"
    echo ""
    echo "Updating train_unsloth.sh with optimal batch size..."

    # Update the training script with the optimal batch size
    sed -i "s/--per_device_train_batch_size [0-9]*/--per_device_train_batch_size ${SUCCESSFUL_BATCH_SIZE}/" train_unsloth.sh

    echo "✓ Updated train_unsloth.sh"
    echo ""
    echo "Ready to train with batch_size=${SUCCESSFUL_BATCH_SIZE}"
else
    echo "✗ All batch sizes failed. Check GPU memory."
fi
echo "========================================================"
