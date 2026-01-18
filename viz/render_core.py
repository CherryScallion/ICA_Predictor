#!/usr/bin/env python3
# viz/render_core.py
"""
可视化模型预测的 3D fMRI 激活图
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import nibabel as nib
import numpy as np
from nilearn import plotting
import os
import matplotlib.pyplot as plt
from utils.paths import get_config_path, get_checkpoint_dir, get_template_dir, resolve_path

# 配置路径 - 使用统一的路径工具
MODEL_PATH = get_checkpoint_dir() / "model_ep50.pth" 
TEMPLATE_DIR = get_template_dir()
OUTPUT_DIR = resolve_path("./results/visuals")

# 如果没有 template，你需要先跑 run_ica 生成它们，或者用代码里 shape 硬算
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Components - 尝试多种可能的文件名
    print("Loading geometry...")
    template_dir = TEMPLATE_DIR
    
    # 尝试找到 ICA 基底文件（rebuild_h5.py 生成的是 ica_mixing_matrix.pt，run_ica.py 生成的是 ica_basis.pt）
    ica_candidates = [
        template_dir / "ica_mixing_matrix.pt",  # rebuild_h5.py 生成
        template_dir / "ica_basis.pt"           # run_ica.py 生成
    ]
    
    # 尝试找到 Mask 文件（rebuild_h5.py 生成的是 mask_dhw.pt，run_ica.py 生成的是 gray_mask.pt）
    mask_candidates = [
        template_dir / "mask_dhw.pt",    # rebuild_h5.py 生成
        template_dir / "gray_mask.pt"    # run_ica.py 生成
    ]
    
    ICA_PATH = None
    for candidate in ica_candidates:
        if candidate.exists():
            ICA_PATH = candidate
            break
    
    MASK_PATH = None
    for candidate in mask_candidates:
        if candidate.exists():
            MASK_PATH = candidate
            break
    
    if ICA_PATH is None or MASK_PATH is None:
        print("❌ 缺少基底文件！")
        print(f"   模板目录: {template_dir}")
        print(f"   需要的文件:")
        print(f"     - ICA基底: ica_mixing_matrix.pt 或 ica_basis.pt")
        print(f"     - Mask: mask_dhw.pt 或 gray_mask.pt")
        print(f"   请运行预处理脚本:")
        print(f"     - python processing/run_ica.py (生成 ica_basis.pt 和 gray_mask.pt)")
        print(f"     - 或 python processing/rebuild_h5.py (生成 ica_mixing_matrix.pt 和 mask_dhw.pt)")
        return
    
    try:
        # 修复 FutureWarning: 使用 weights_only=False（默认值，但显式指定以避免警告）
        basis = torch.load(ICA_PATH, map_location=device, weights_only=False)  # [K, Voxels] 或 [K, D, H, W]
        # Mask 需要转换为 numpy，所以加载到 CPU
        mask_tensor = torch.load(MASK_PATH, map_location='cpu', weights_only=False)
        
        # 处理不同的 mask 格式（确保在 CPU 上才能转换为 numpy）
        if mask_tensor.dim() == 3:  # [D, H, W]
            mask_np = mask_tensor.numpy().astype(bool)
        elif mask_tensor.dim() == 4:  # [1, D, H, W] 或类似
            mask_np = mask_tensor.squeeze().numpy().astype(bool)
        else:
            mask_np = mask_tensor.numpy().astype(bool)
        
        # 处理不同的 basis 格式
        # rebuild_h5.py: [K, Voxels] (扁平)
        # run_ica.py: 可能是 [K, D, H, W] 或其他格式
        if basis.dim() == 2 and basis.shape[1] != mask_np.sum():
            # 如果 basis 是 [K, Voxels] 格式，需要与 mask 配合使用
            print(f"✅ Loaded ICA basis: {basis.shape} (flattened format)")
            print(f"✅ Loaded mask: {mask_np.shape}, {mask_np.sum()} voxels")
        else:
            print(f"✅ Loaded ICA basis: {basis.shape}")
            print(f"✅ Loaded mask: {mask_np.shape}, {mask_np.sum()} voxels")
            
    except Exception as e:
        print(f"❌ 加载基底文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load Model & Data
    from data.loaders import FMRIEEGDataset
    from models.classifier_net import PhysicsE2fNet
    import yaml
    
    # 获取一条真实数据来测试（使用配置文件）
    config_path = get_config_path()
    ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=False)
    sample_eeg, sample_gt = ds[0]  # 取第0个样本 [20, 64, 249], [64]
    
    # Load Model (使用配置文件中的参数)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    eeg_c, eeg_f, eeg_t = sample_eeg.shape
    model = PhysicsE2fNet(
        n_ica_components=cfg['data_specs']['n_ica_components'],
        eeg_channels=eeg_c,
        eeg_time_len=eeg_t,
        basis_path=str(ICA_PATH) if ICA_PATH.exists() else None,
        task='regression'
    ).to(device)
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        print(f"✅ Model loaded from: {MODEL_PATH}")
    else:
        print(f"⚠️  Warning: Model file not found at {MODEL_PATH}")
        print("   Using untrained model for demonstration.")
    model.eval()
    
    # 3. Predict
    print("Predicting...")
    input_tensor = sample_eeg.unsqueeze(0).to(device).float()
    with torch.no_grad():
        # PhysicsE2fNet forward 里面如果 task='regression' 直接返回 weights
        pred_weights = model(input_tensor) # [1, 64]
    
    # 4. Rendering Logic (Inverse Projection)
    print("Composing 3D Volume...")
    
    # 处理不同的 basis 格式
    # 确保 basis 与 pred_weights 在同一设备上（通常都在 CUDA）进行矩阵乘法
    if not isinstance(basis, torch.Tensor):
        basis = torch.from_numpy(basis)
    
    # 将 basis 移到与 pred_weights 相同的设备
    basis = basis.to(pred_weights.device)
    
    if basis.dim() == 2:
        # 格式1: [K, Voxels] (扁平格式，rebuild_h5.py 生成)
        # [1, 64] @ [64, Voxels] -> [1, Voxels]
        pred_vec = pred_weights @ basis
        pred_vec = pred_vec.cpu().numpy().flatten()
    elif basis.dim() == 4:
        # 格式2: [K, D, H, W] (空间格式，run_ica.py 可能生成)
        # 需要将 basis 展平并与 mask 配合使用
        K, D, H, W = basis.shape
        basis_flat = basis.reshape(K, -1)  # [K, D*H*W]
        
        # [1, K] @ [K, D*H*W] -> [1, D*H*W]
        pred_vec_flat = pred_weights @ basis_flat
        pred_vec = pred_vec_flat.cpu().numpy().flatten()
    else:
        raise ValueError(f"Unsupported basis format: {basis.shape}")
    
    # 由于没有归一化，数值可能很大 (-200~200)，我们需要 Clip 一下防止画图炸掉
    # 或者画图软件会自动归一化，我们先打印一下范围
    print(f"Activation Range: min={pred_vec.min():.2f}, max={pred_vec.max():.2f}")
    
    # Unmasking -> [D, H, W]
    vol = np.zeros(mask_np.shape)
    vol[mask_np] = pred_vec
    
    # Transpose [D, H, W] -> [X, Y, Z] for Nifti (Subject to trial)
    # 假设 mask 是 [30, 64, 64]，Nifti 通常喜欢 [64, 64, 30]
    vol_nii = np.transpose(vol, (2, 1, 0)) 
    
    # Save
    nii_img = nib.Nifti1Image(vol_nii, affine=np.eye(4)) # Fake affine
    save_f = str(OUTPUT_DIR / "prediction_sample0.nii.gz")
    nib.save(nii_img, save_f)
    print(f"✅ 3D Volume saved to: {save_f}")
    
    # 5. Quick Plot (Slices)
    print("Generating Slice Plot...")
    fig = plt.figure(figsize=(10, 3))
    plotting.plot_stat_map(
        nii_img, 
        bg_img=None, # 黑背景
        display_mode='ortho', 
        cut_coords=(32, 32, 15), # 切个中心点看看
        threshold=10.0, # 设置一个绝对值阈值过滤底噪
        figure=fig,
        title="Predicted Activation"
    )
    slice_view_path = str(OUTPUT_DIR / "slice_view.png")
    fig.savefig(slice_view_path)
    print("✅ PNG Snapshot saved.")

if __name__ == "__main__":
    main()