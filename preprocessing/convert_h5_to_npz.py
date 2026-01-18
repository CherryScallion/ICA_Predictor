#!/usr/bin/env python3
"""
Convert H5 data to optimized NPZ format for faster DataLoader performance.
This eliminates H5 I/O overhead by precomputing and storing in numpy-compatible format.
"""

import h5py
import numpy as np
import os
from pathlib import Path
import glob

def normalize_eeg_freq(eeg_data, target_bins=64):
    """Normalize EEG frequency dimension to target size."""
    N, C, F, T = eeg_data.shape
    
    if F == target_bins:
        return eeg_data
    elif F < target_bins:
        pad_width = ((0, 0), (0, 0), (0, target_bins - F), (0, 0))
        return np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
    else:
        return eeg_data[:, :, :target_bins, :]

def convert_h5_to_npz(h5_path, npz_output_dir, target_freq_bins=64):
    """
    Convert single H5 file to NPZ format.
    
    Args:
        h5_path: Path to H5 file
        npz_output_dir: Output directory for NPZ files
        target_freq_bins: Target frequency bins for EEG
    """
    os.makedirs(npz_output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as hf:
        eeg_data = hf['eeg'][:]  # [N, C, F, T]
        weight_data = hf['weights'][:]  # [N, 64]
        
        # Normalize frequency dimension
        eeg_data = normalize_eeg_freq(eeg_data, target_freq_bins)
        
        # Convert to float32 if needed
        eeg_data = eeg_data.astype(np.float32)
        weight_data = weight_data.astype(np.float32)
        
        # Save as NPZ
        npz_name = Path(h5_path).stem + '.npz'
        npz_path = os.path.join(npz_output_dir, npz_name)
        
        np.savez_compressed(
            npz_path,
            eeg=eeg_data,
            weights=weight_data
        )
        
        print(f"Converted: {h5_path}")
        print(f"  EEG shape: {eeg_data.shape}")
        print(f"  Weights shape: {weight_data.shape}")
        print(f"  Saved to: {npz_path}")
        print()

def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.paths import resolve_path
    
    # 使用配置文件中的路径，如果没有则使用默认路径
    h5_dir = str(resolve_path('./data/H5'))
    npz_dir = str(resolve_path('./data/optimized_npz'))
    
    # Find all H5 files
    h5_files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
    
    print(f"Found {len(h5_files)} H5 files to convert")
    print(f"Output directory: {npz_dir}\n")
    
    for h5_path in h5_files:
        convert_h5_to_npz(h5_path, npz_dir)
    
    print(f"Conversion complete! Total files: {len(h5_files)}")

if __name__ == '__main__':
    main()
