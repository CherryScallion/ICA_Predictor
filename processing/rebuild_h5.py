import os
import glob
import sys
import yaml
from pathlib import Path
import h5py
import numpy as np
import torch
import nibabel as nib
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from nilearn.image import new_img_like
from nilearn.masking import compute_epi_mask, apply_mask

# 引用你的配置，保证路径统一
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import get_config_path, resolve_path

def load_config():
    """Load train_config.yaml using unified path resolution."""
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find train_config.yaml at: {config_path}\n"
            f"You can set HACHIMINET_ROOT env var to point to the project root."
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CFG = load_config()
RAW_DIR = str(resolve_path(CFG['paths']['raw_h5_dir']))         # 原数据
OPT_DIR = str(resolve_path(CFG['paths'].get('optimized_h5_dir', './data/H5')))  # 新数据存放处 (压缩后)
TEMPLATE_DIR = str(resolve_path(CFG['paths']['template_dir']))  # 存放 Mask 和 Basis (视觉基准)
N_COMPONENTS = 64

os.makedirs(OPT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def step1_fit_global_ica(files):
    print("--- Step 1: 拟合 Global ICA 基底 ---")
    
    # 策略：为了避免内存爆炸，我们不加载所有数据。
    # 我们随机抽取若干个样本 (比如总共 1000 个 snapshot) 来训练 ICA。
    # NODDI 有 274 samples, Oddball 有 144 samples.
    # 我们可以把前 5 个 H5 的所有数据拼起来训练。
    
    # Determine maximum depth across all files to handle variable slice counts
    max_depth = 0
    for fp in files:
        try:
            with h5py.File(fp, 'r') as hf:
                d = hf['fmri'].shape[1]
                if d > max_depth:
                    max_depth = d
        except Exception:
            continue

    # Sample a small subset for ICA training, but pad volumes to max_depth
    subset_files = files[:5]
    data_accumulator = []

    affine = np.eye(4) # 伪坐标系，用于 Nilearn

    print(f"Loading subset from {len(subset_files)} files (padded to depth={max_depth})...")
    for fp in subset_files:
        with h5py.File(fp, 'r') as hf:
            # fMRI shape: [N_samples, Depth, H, W]
            data = hf['fmri'][:]
            D = data.shape[1]
            if D < max_depth:
                pad_width = ((0, 0), (0, max_depth - D), (0, 0), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
            data_accumulator.append(data)
    
    # Concat -> [Total_Samples, 30, 64, 64]
    full_data = np.concatenate(data_accumulator, axis=0)
    print(f"ICA Training Data Shape: {full_data.shape}")
    
    # 1. 计算全局 Mask
    # 将所有时间点平均，得到一张平均脑图
    mean_vol = np.mean(full_data, axis=0) # [30, 64, 64]
    
    # 注意 Nilearn compute_epi_mask 期望的是 [X, Y, Z]。
    # 你的数据看起来是 [Depth, H, W]。保险起见，我们将维度转置一下喂给 Nilearn
    # Transpose [D, H, W] -> [H, W, D] (即 64, 64, max_depth)
    mean_vol_nii = nib.Nifti1Image(np.transpose(mean_vol, (1, 2, 0)), affine)
    mask_img = compute_epi_mask(mean_vol_nii)
    mask_bool = mask_img.get_fdata() > 0 # [64, 64, 30]
    
    # 为了后续方便，我们将 mask 转置回 [D, H, W] 以匹配原数据顺序
    mask_bool_dhw = np.transpose(mask_bool, (2, 0, 1)) # [30, 64, 64]
    
    # 保存 Mask (用于可视化)
    torch.save(torch.tensor(mask_bool_dhw), os.path.join(TEMPLATE_DIR, 'mask_dhw.pt'))
    print(f"Mask computed and saved. Voxels: {np.sum(mask_bool_dhw)}")
    
    # 2. Masking (Flattening)
    # data: [N, D, H, W], mask: [D, H, W]
    # Result -> [N, n_voxels]
    masked_data = full_data[:, mask_bool_dhw] 
    
    # 3. FastICA
    print(f"Fitting FastICA (K={N_COMPONENTS})...")
    ica = FastICA(n_components=N_COMPONENTS, random_state=42, whiten='unit-variance')
    
    # 拟合：sklearn 的 FastICA 期望 [n_samples, n_features]
    # 我们这里 samples 是行，voxels 是特征，刚好。
    # 输出 sources: [n_samples, n_components] (这里的 sources 是 temporal/sample weights)
    # 得到的 mixing_: [n_voxels, n_components] ? 还是 components_: [n_components, n_voxels]
    
    S_weights = ica.fit_transform(masked_data)  # [N, K] (这是样本在这个空间的表达)
    A_maps = ica.components_                # [K, n_voxels] (这是空间基底!)
    
    # 4. 保存基底 (Spatial Basis)
    # 这就是我们要存下来的"透明胶片"，用来可视化
    print(f"ICA Basis Shape: {A_maps.shape}") # [64, 46900] roughly
    torch.save(torch.tensor(A_maps), os.path.join(TEMPLATE_DIR, 'ica_mixing_matrix.pt'))
    
    # 同时也保存一个 inverse_transform 用的伪逆矩阵，用于投影
    # W = Y * A_pinv
    A_pinv = np.linalg.pinv(A_maps) # [n_voxels, K]
    
    return mask_bool_dhw, A_maps, A_pinv

def step2_rebuild_dataset(files, mask_bool, basis_maps, basis_pinv):
    print("\n--- Step 2: 重建数据集 (Rebuilding H5) ---")
    
    # 这一步我们把原本几十 G 的 fMRI，替换为只有几 MB 的 Weights
    
    basis_maps_t = basis_maps.T # [Voxels, K]
    
    for fpath in files:
        fname = os.path.basename(fpath)
        save_path = os.path.join(OPT_DIR, f"opt_{fname}")
        
        print(f"Processing {fname} -> opt_{fname}")
        
        with h5py.File(fpath, 'r') as src, h5py.File(save_path, 'w') as dst:
            # 1. 搬运 EEG (不做改变)
            if 'eeg' in src:
                # Direct copy, or simple verify shape
                # eeg_data: [N, 20, 64, 249]
                eeg_data = src['eeg'][:]
                dst.create_dataset('eeg', data=eeg_data, compression="gzip")
            else:
                print(f"⚠️ Warning: No 'eeg' in {fname}")
                continue
                
            # 2. 压缩 fMRI -> Weights
            if 'fmri' in src:
                fmri_data = src['fmri'][:] # [N, D, H, W]
                N = fmri_data.shape[0]

                # Ensure fmri_data depth matches mask depth (pad or trim as needed)
                mask_depth = mask_bool.shape[0]
                D = fmri_data.shape[1]
                if D < mask_depth:
                    pad_width = ((0, 0), (0, mask_depth - D), (0, 0), (0, 0))
                    fmri_data = np.pad(fmri_data, pad_width, mode='constant', constant_values=0)
                elif D > mask_depth:
                    fmri_data = fmri_data[:, :mask_depth, :, :]

                # Apply Mask -> [N, n_voxels]
                flat_data = fmri_data[:, mask_bool]
                
                # Projection / Dual Regression
                # 我们要解出 Y = W * Maps
                # 已知 Y [N, Vox], Maps [K, Vox]
                # W = Y * Maps_pinv 
                # 或 W = Y * Maps.T (如果假设正交，但这比较粗糙)
                # 使用我们之前算好的 pinv 最准: [Vox, K]
                
                weights = np.dot(flat_data, basis_pinv) # [N, K]
                
                # Z-Score Normalization of Weights (Optional but recommended for regression)
                # 最好在 global 做，但在这里 run-wise 也可以，只要保持统一
                # 为了保持简单，先存储原始 projection 权重
                
                dst.create_dataset('weights', data=weights, compression="gzip")
                
                # 存一个 shape 属性，以此证明转换了多少倍
                ratio = flat_data.nbytes / weights.nbytes
                print(f"   Shape: fMRI {flat_data.shape} -> Weights {weights.shape}")
                print(f"   Compression Ratio: {ratio:.1f}x")
            
            # Copy other attributes if any
            for attr in src.attrs:
                dst.attrs[attr] = src.attrs[attr]

def main():
    h5_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.h5")))
    if not h5_files:
        print("No H5 files found.")
        return
        
    # Phase 1: Learn Anatomy from a subset
    mask, basis, pinv = step1_fit_global_ica(h5_files)
    
    # Phase 2: Convert everything
    step2_rebuild_dataset(h5_files, mask, basis, pinv)
    
    print("\n✅ 所有数据处理完毕！")
    print(f"训练用数据位于: {OPT_DIR}")
    print("DataLoader 请改为直接读取这里的 .h5 文件，Key 为 'eeg' 和 'weights'。")

if __name__ == "__main__":
    main()