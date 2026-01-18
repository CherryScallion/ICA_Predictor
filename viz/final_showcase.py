#!/usr/bin/env python3
"""
最终展示可视化脚本 - 生成预测与真实值的对比图
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image
import warnings
from utils.paths import get_config_path, get_checkpoint_dir, get_template_dir, resolve_path

# 忽略 nilearn 的一些 warning 以保持清爽
warnings.filterwarnings("ignore")

# --- 配置区 - 使用统一的路径工具 ---
MODEL_PATH = get_checkpoint_dir() / "model_ep50.pth"
TEMPLATE_DIR = get_template_dir()
OUTPUT_DIR = resolve_path("./results/final_showcase")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def vector_to_nifti(weights, ica_basis, mask_bool, affine=np.eye(4)):
    """将 64维权重 -> 3D NIfTI 图像"""
    # 1. 逆投影: [1, 64] @ [64, Voxels] -> [1, Voxels]
    # 转换为 torch.Tensor 并确保在同一设备上
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    elif isinstance(weights, torch.Tensor):
        weights = weights.detach()
    
    if isinstance(ica_basis, np.ndarray):
        ica_basis = torch.from_numpy(ica_basis)
    
    # 确保 weights 和 ica_basis 在同一设备上进行矩阵乘法
    if isinstance(ica_basis, torch.Tensor):
        ica_basis = ica_basis.to(weights.device)
    
    # 执行矩阵乘法（在同一设备上）
    activation_vec_tensor = torch.matmul(weights, ica_basis)
    # 移回 CPU 并转换为 numpy
    activation_vec = activation_vec_tensor.cpu().numpy().flatten()
    
    # 2. 填入 3D 空间
    vol_data = np.zeros(mask_bool.shape)
    vol_data[mask_bool] = activation_vec
    
    # 3. 转置适配 NIfTI (通常 D,H,W -> X,Y,Z 可能会有轴变换，视 nilearn 习惯)
    # 我们之前的脚本是把 depth 放第一位的 [30, 64, 64]，Nifti 习惯 [64, 64, 30]
    vol_nii_data = np.transpose(vol_data, (2, 1, 0)) 
    
    return nib.Nifti1Image(vol_nii_data, affine)

def main():
    print(f"🌟 Starting Final Showcase Generation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载组件 - 尝试多种可能的文件名
    template_dir = TEMPLATE_DIR
    
    # 尝试找到 ICA 基底文件
    ica_candidates = [
        template_dir / "ica_mixing_matrix.pt",
        template_dir / "ica_basis.pt"
    ]
    
    # 尝试找到 Mask 文件
    mask_candidates = [
        template_dir / "mask_dhw.pt",
        template_dir / "gray_mask.pt"
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
        print("❌ Error: 找不到基底文件。")
        print(f"   模板目录: {template_dir}")
        print(f"   需要的文件:")
        print(f"     - ICA基底: ica_mixing_matrix.pt 或 ica_basis.pt")
        print(f"     - Mask: mask_dhw.pt 或 gray_mask.pt")
        print(f"   请运行预处理脚本:")
        print(f"     - python processing/run_ica.py")
        print(f"     - 或 python processing/rebuild_h5.py")
        return
    
    try:
        basis = torch.load(ICA_PATH, map_location=device, weights_only=False)
        mask_tensor = torch.load(MASK_PATH, map_location='cpu', weights_only=False)
        
        # 处理不同的 mask 格式
        if mask_tensor.dim() == 3:
            mask = mask_tensor.numpy().astype(bool)
        elif mask_tensor.dim() == 4:
            mask = mask_tensor.squeeze().numpy().astype(bool)
        else:
            mask = mask_tensor.numpy().astype(bool)
            
        print(f"✅ Loaded ICA basis: {basis.shape}")
        print(f"✅ Loaded mask: {mask.shape}, {mask.sum()} voxels")
    except Exception as e:
        print(f"❌ Error loading template files: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 准备标准解剖底图 (MNI Template)
    # 自动下载 MNI152 标准脑，以此作为背景，显得专业
    print("Fetching MNI Template for background...")
    mni_template = datasets.load_mni152_template()

    # 3. 加载模型
    from models.classifier_net import PhysicsE2fNet
    import yaml
    
    # 从配置文件获取参数
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 先获取一条数据来确定维度
    from data.loaders import FMRIEEGDataset
    ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=False)
    sample_eeg, _ = ds[0]
    eeg_c, eeg_f, eeg_t = sample_eeg.shape
    
    model = PhysicsE2fNet(
        n_ica_components=cfg['data_specs']['n_ica_components'],
        eeg_channels=eeg_c,
        eeg_time_len=eeg_t,
        basis_path=str(ICA_PATH) if ICA_PATH.exists() else None,
        task='regression'
    ).to(device)
    
    # 加载权重
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from: {MODEL_PATH}")
    else:
        print(f"⚠️  Warning: Model file not found at {MODEL_PATH}")
        print("   Using untrained model for demonstration.")
    model.eval()
    print("✅ Model loaded successfully.")

    # 4. 获取数据（已在上面加载）
    
    # 5. 开始生成对比图
    # 随机取 3 个样本展示
    indices_to_show = [0, 50, 100] # 可以改随机
    indices_to_show = [i for i in indices_to_show if i < len(ds)]
    
    print(f"Generating visualizations for samples: {indices_to_show}")
    
    for idx in indices_to_show:
        eeg, gt_weights = ds[idx]
        
        # 预测
        eeg_input = eeg.unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred_weights = model(eeg_input) # [1, 64]
            
        # --- 生成 3D 图像 ---
        img_pred = vector_to_nifti(pred_weights, basis, mask)
        img_gt = vector_to_nifti(gt_weights.unsqueeze(0), basis, mask)
        
        # 由于我们用的坐标系是 Fake Affine，为了叠加到 MNI 上，需要 Resample
        # 这步是关键：把我们的 [64,64,30] 强行插值对齐到 [91,109,91] 的 MNI
        # 这样看起来才是在“真正的大脑”上
        img_pred_resampled = image.resample_to_img(img_pred, mni_template)
        img_gt_resampled = image.resample_to_img(img_gt, mni_template)
        
        # --- 绘图 ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 为了美观，设置阈值，把那种接近0的背景底噪切掉
        # 你的 loss 很大，说明数值很大 (比如 +-100)，那阈值可以设为 max 的 20%
        display_threshold = np.max(np.abs(pred_weights.cpu().numpy())) * 0.2
        
        # Plot 1: Prediction
        plotting.plot_stat_map(
            img_pred_resampled, bg_img=mni_template, 
            display_mode='z', cut_coords=5, # Z轴切5张
            threshold=display_threshold,
            title=f"Sample {idx} - EEG Prediction",
            axes=axes[0], colorbar=True
        )
        
        # Plot 2: Ground Truth
        plotting.plot_stat_map(
            img_gt_resampled, bg_img=mni_template, 
            display_mode='z', cut_coords=5, 
            threshold=display_threshold, # 保持阈值一致以便对比
            title=f"Sample {idx} - Ground Truth (fMRI)",
            axes=axes[1], colorbar=True
        )
        
        save_p = str(OUTPUT_DIR / f"comparison_sample_{idx}.png")
        plt.savefig(save_p)
        plt.close()
        print(f"   Saved comparison: {save_p}")

    print("\n🎉 Done! 请打开 results/final_showcase 文件夹查看最终成果！")

if __name__ == "__main__":
    main()