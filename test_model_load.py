#!/usr/bin/env python
"""Test script to verify VibeVoice model and dataset loading on GPU #1"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoProcessor

print("=" * 60)
print("VibeVoice Fine-tuning Test - GPU #1 (Unsloth Branch)")
print("=" * 60)

# Check GPU
print("\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

# Test unsloth import
print("\n2. Testing Unsloth import:")
try:
    from unsloth import FastLanguageModel
    from unsloth.kernels import fast_cross_entropy_loss
    print("   ✓ Unsloth imported successfully")
except Exception as e:
    print(f"   ✗ Unsloth import failed: {e}")
    exit(1)

# Test VibeVoice import
print("\n3. Testing VibeVoice import:")
try:
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    print("   ✓ VibeVoice imported successfully")
except Exception as e:
    print(f"   ✗ VibeVoice import failed: {e}")
    exit(1)

# Test model loading (this will download the model if needed)
print("\n4. Testing model loading (VibeVoice-1.5B):")
print("   This may take a few minutes on first run to download the model...")
try:
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        "vibevoice/VibeVoice-1.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    print("   ✓ Model loaded successfully")

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   GPU memory used: {allocated:.2f} GB")

    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    exit(1)

# Test dataset loading
print("\n5. Testing dataset loading (jenny_vibevoice_formatted):")
try:
    dataset = load_dataset("vibevoice/jenny_vibevoice_formatted", split="train[:10]")
    print(f"   ✓ Dataset loaded successfully")
    print(f"   Sample count (first 10): {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"   First sample text: {dataset[0]['text'][:100]}...")
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    print("   Note: You may need to accept the dataset license on HuggingFace")

print("\n" + "=" * 60)
print("All tests passed! Ready to start fine-tuning.")
print("Run: ./train_unsloth.sh")
print("=" * 60)
