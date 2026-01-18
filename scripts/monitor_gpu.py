#!/usr/bin/env python3
"""
Monitor GPU memory and usage during training.
Run in separate terminal while training is ongoing.
"""

import torch
import time
import subprocess
import sys

def monitor_gpu(interval=2, max_duration=3600):
    """Monitor GPU memory allocation."""
    
    print("GPU Monitor Started")
    print("=" * 60)
    print(f"{'Time':>8} | {'Allocated(GB)':>14} | {'Reserved(GB)':>14} | {'Free(GB)':>10}")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            break
            
        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - allocated
            
            print(f"{elapsed:8.1f}s | {allocated:14.2f} | {reserved:14.2f} | {free:10.2f}")
            sys.stdout.flush()
            
            time.sleep(interval)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print("=" * 60)
    print("GPU Monitor Stopped")

if __name__ == '__main__':
    monitor_gpu(interval=3)
