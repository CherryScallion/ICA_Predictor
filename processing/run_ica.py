# processing/run_ica.py
import os
import sys
import yaml
import h5py
import torch
import numpy as np
import nibabel as nib
from nilearn import decomposition
from nilearn.masking import compute_epi_mask
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import get_config_path, resolve_path

def run_custom_ica():
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        CFG = yaml.safe_load(f)

    # 1. 搜集前几个 H5，准备数据
    H5_FMRI_KEY = CFG['data_specs']['h5_key_fmri']
    raw_h5_dir = str(resolve_path(CFG['paths']['raw_h5_dir']))
    files = sorted(glob.glob(os.path.join(raw_h5_dir, "*.h5")))[:30] # 只用30个样本做基底
    
    nifti_list = []
    
    # 我们需要构建一个“假”的 Affine 矩阵，因为没有原始头文件
    # 假设是标准各向同性体素 (这里不影响深度学习，只影响可视化)
    fake_affine = np.eye(4) 
    
    first_subject_data = None # 用于生成 Mask

    print(f"Loading data for ICA from {len(files)} files...")
    for fpath in files:
        with h5py.File(fpath, 'r') as hf:
            # 数据形状: [N, D, H, W] -> (Samples, 30, 64, 64)
            data = hf[H5_FMRI_KEY][:]
            
            # Nilearn 需要: [X, Y, Z, Time]
            # 我们假设 (D, H, W) 对应 Z, Y, X 或 X, Y, Z
            # 转置为: [H, W, D, N] -> (64, 64, 30, Samples)
            data_permuted = np.transpose(data, (2, 3, 1, 0))
            
            if first_subject_data is None:
                first_subject_data = data_permuted

            # 封装为 Nifti
            img = nib.Nifti1Image(data_permuted, fake_affine)
            nifti_list.append(img)

    # 2. 从数据自动计算 Mask (Masking Strategy)
    print("Computing brain mask from data (Compute EPI Mask)...")
    # 对第一个受试者求平均图
    mean_img = nib.Nifti1Image(np.mean(first_subject_data, axis=-1), fake_affine)
    # 计算 mask
    mask_img = compute_epi_mask(mean_img)
    
    # 保存 Mask
    mask_data = mask_img.get_fdata() > 0
    template_dir = str(resolve_path(CFG['paths']['template_dir']))
    os.makedirs(template_dir, exist_ok=True)
    torch.save(torch.tensor(mask_data), os.path.join(template_dir, 'gray_mask.pt'))
    print(f"Mask Generated: {mask_data.shape}")

    # 3. 运行 CanICA
    print(f"Running CanICA (K={CFG['data_specs']['n_ica_components']})...")
    canica = decomposition.CanICA(
        n_components=CFG['data_specs']['n_ica_components'],
        mask=mask_img,       # 使用自动计算的 Mask
        smoothing_fwhm=None, # 数据已经是processed可能不需要大平滑
        n_jobs=-1,
        verbose=1
    )
    
    canica.fit(nifti_list)
    
    # [X, Y, Z, K] -> [K, X, Y, Z]
    components = canica.components_img_.get_fdata()
    basis_data = np.moveaxis(components, -1, 0) # 转置 Channel first
    
    # 还要把维度转回去匹配 PyTorch 这里的 D, H, W (Z, Y, X?)
    # 之前是 (H, W, D, N), Nilearn 也是按这个 fit 的
    # 这里我们认为 PyTorch 需要 (K, D, H, W)，所以需要留意轴的顺序
    # 简单处理：CanICA输出空间维跟输入一致，输入是 H,W,D，输出就是 H,W,D
    # basis: [K, H, W, D]. Permute -> [K, D, H, W]
    basis_data = np.transpose(basis_data, (0, 3, 1, 2))
    
    os.makedirs(template_dir, exist_ok=True)
    torch.save(torch.tensor(basis_data, dtype=torch.float32), 
               os.path.join(template_dir, 'ica_basis.pt'))
    print("ICA Basis Saved.")

if __name__ == "__main__":
    run_custom_ica()