#!/usr/bin/env python
"""Test script to verify VibeVoice-7B model loading on GPU #1 with 24GB VRAM"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoProcessor

print("=" * 60)
print("VibeVoice-7B Loading Test - GPU #1 (24GB VRAM)")
print("=" * 60)

# Check GPU
print("\n1. GPU Status:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

# Test VibeVoice import
print("\n2. Importing VibeVoice...")
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
print("   ✓ Import successful")

# Test 7B model loading with memory optimizations
print("\n3. Loading VibeVoice-7B model:")
print("   This will take several minutes to download (~13GB)...")
print("   Loading with BF16 precision to reduce memory...")

try:
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        "vibevoice/VibeVoice-7B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    print("   ✓ Model loaded successfully!")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
        print(f"\n   GPU Memory Status:")
        print(f"   - Allocated: {allocated:.2f} GB")
        print(f"   - Reserved: {reserved:.2f} GB")
        print(f"   - Free: {free:.2f} GB")

    # Test if we can apply LoRA
    print("\n4. Testing LoRA configuration:")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Check trainable params after LoRA
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params before LoRA: {trainable_before:,}")

    print("   Note: LoRA will be applied during training")

    # Clean up
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("✓ 7B model test passed! Ready for fine-tuning.")
    print("=" * 60)
    print("\nMemory optimizations enabled:")
    print("  • BF16 precision")
    print("  • Gradient checkpointing")
    print("  • Batch size = 1")
    print("  • LoRA (low-rank adaptation)")
    print("\nRun: ./train_unsloth.sh")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    print("\nIf you see OOM (Out of Memory) error, we may need to:")
    print("  1. Use smaller LoRA rank (r=4)")
    print("  2. Disable diffusion head training")
    print("  3. Apply LoRA to diffusion head instead of full training")
    exit(1)
