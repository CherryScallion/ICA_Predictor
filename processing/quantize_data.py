# processing/quantize_data.py
import os
import sys
import yaml
import h5py
import torch
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import get_config_path, resolve_path

with open(get_config_path(), 'r') as f:
    CFG = yaml.safe_load(f)

def generate_hachimi_labels():
    raw_dir = str(resolve_path(CFG['paths']['raw_h5_dir']))
    save_dir = str(resolve_path(CFG['paths']['processed_dir']))
    template_dir = str(resolve_path(CFG['paths']['template_dir']))
    mask_path = os.path.join(template_dir, 'gray_mask.pt')
    
    os.makedirs(save_dir, exist_ok=True)
    mask_bool = torch.load(mask_path).numpy().astype(bool) # [D, H, W] 或 [H, W, D] 注意上面脚本的一致性

    # 简单 Hack：确保 mask 形状和数据 fmri[0] 形状匹配
    # 如果不匹配需要在此处调整 mask 的 transpose
    
    files = sorted(glob.glob(os.path.join(raw_dir, "*.h5")))
    thresh = CFG['data_specs']['activation_threshold']
    
    for fpath in files:
        fname = os.path.basename(fpath).replace('.h5', '')
        
        with h5py.File(fpath, 'r') as hf:
            # 1. EEG: [N, C, H, W] -> 保存 [N, C, H, W]
            eeg_data = hf['eeg'][:]
            # 注意: 如果 Encoder 还是按照 Sequence 处理, N 应该是 Time.
            torch.save(torch.tensor(eeg_data, dtype=torch.float32), 
                       os.path.join(save_dir, f"{fname}_eeg.pt"))
            
            # 2. fMRI: [N, D, H, W] -> (Samples, 30, 64, 64)
            fmri_data = hf['fmri'][:]
            
            N = fmri_data.shape[0]
            # [N, Masked_Voxels]
            # 需要先将 fmri_data[i] 根据 Mask 展平
            # mask_bool: [30, 64, 64]
            # broadcast mask: fmri_data [:, mask] works if dimensions align
            
            try:
                # fmri_data 后面三维是空间, 对应 mask
                masked_data = fmri_data[:, mask_bool]
            except IndexError:
                # 如果 run_ica 中转置导致 mask 形状不符 (比如 mask是 [64,64,30])
                # 这里要做相应调整，以数据为准
                if mask_bool.shape != fmri_data.shape[1:]:
                     mask_bool = mask_bool.transpose(2, 0, 1) # 猜一个转置 [30, 64, 64]
                masked_data = fmri_data[:, mask_bool]

            # Z-Score
            scaler = StandardScaler()
            z_data = scaler.fit_transform(masked_data)
            
            # Labeling
            labels = np.zeros_like(z_data, dtype=np.int8)
            labels[z_data > thresh] = 1
            labels[z_data < -thresh] = 2
            
            # Reconstruct to [N, D, H, W]
            full_vol = np.full(fmri_data.shape, -1, dtype=np.int8)
            full_vol[:, mask_bool] = labels
            
            torch.save(torch.tensor(full_vol), os.path.join(save_dir, f"{fname}_labels.pt"))
            print(f"Processed {fname}")

if __name__ == "__main__":
    generate_hachimi_labels()