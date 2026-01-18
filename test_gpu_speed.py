#!/usr/bin/env python3
"""
Direct test: verify that GPU computations are actually happening.
Compare model forward pass speed on CPU vs GPU.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import time
from models.classifier_net import PhysicsE2fNet
from utils.paths import get_template_dir

# Create dummy data
batch_size = 32
eeg_c, eeg_f, eeg_t = 20, 64, 249

# Use path utilities for basis path
basis_path = get_template_dir() / 'ica_mixing_matrix.pt'
model = PhysicsE2fNet(
    n_ica_components=64,
    eeg_channels=eeg_c,
    eeg_time_len=eeg_t,
    basis_path=str(basis_path) if basis_path.exists() else None
)

dummy_input = torch.randn(batch_size, eeg_c, eeg_f, eeg_t)

# Test 1: CPU execution
print("=" * 60)
print("CPU Test")
print("=" * 60)
model.to('cpu')
model.eval()

with torch.no_grad():
    start = time.time()
    for _ in range(10):
        _ = model(dummy_input.to('cpu'))
    cpu_time = time.time() - start

print(f"CPU Time (10 iterations): {cpu_time:.2f}s -> {cpu_time/10*1000:.1f}ms per iteration")

# Test 2: GPU execution
print("\n" + "=" * 60)
print("GPU Test")
print("=" * 60)
model.to('cuda')
dummy_input_gpu = dummy_input.to('cuda')

with torch.no_grad():
    torch.cuda.synchronize()  # Wait for GPU to be ready
    start = time.time()
    for _ in range(10):
        output = model(dummy_input_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start

print(f"GPU Time (10 iterations): {gpu_time:.2f}s -> {gpu_time/10*1000:.1f}ms per iteration")

print(f"\n[Result] GPU Speedup: {cpu_time/gpu_time:.1f}x")
print(f"[Result] GPU Allocated Memory: {torch.cuda.memory_allocated()/1e9:.3f}GB")
