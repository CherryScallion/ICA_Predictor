#!/usr/bin/env python3
"""
可视化模型预测结果 - 检查训练是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data.loaders import FMRIEEGDataset
from models.classifier_net import PhysicsE2fNet
from utils.paths import get_config_path, get_checkpoint_dir

# 配置路径
MODEL_PATH = get_checkpoint_dir() / "model_ep50.pth"  # 加载刚训好的模型

def visualize_prediction():
    # 1. 加载数据 (使用配置文件)
    config_path = get_config_path()
    ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=False)
    
    # 取最后的几个样本，假设它们是验证集
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. 获取数据形状以初始化模型
    sample_eeg, _ = ds[0]
    eeg_c, eeg_f, eeg_t = sample_eeg.shape
    print(f"EEG shape: C={eeg_c}, F={eeg_f}, T={eeg_t}")
    
    # 加载模型 (使用与训练时相同的模型结构)
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    model = PhysicsE2fNet(
        n_ica_components=cfg['data_specs']['n_ica_components'],
        eeg_channels=eeg_c,
        eeg_time_len=eeg_t,
        task='regression'
    ).to(device)
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        print(f"⚠️  Warning: Model file not found at {MODEL_PATH}")
        print("   Using untrained model for demonstration.")
    
    model.eval()
    
    # 3. 预测
    eeg, target = next(iter(dl))
    eeg = eeg.to(device).float()
    target = target.to(device).float()
    
    with torch.no_grad():
        pred = model(eeg)  # [B, 64]
        
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # 4. 画图对比
    plt.figure(figsize=(15, 6))
    
    # 选取第 0 个样本进行对比
    sample_idx = 0
    
    # 绘制 Component 权重对比 (64个维度的分布)
    plt.subplot(1, 2, 1)
    plt.plot(target[sample_idx], label='Ground Truth', color='blue', alpha=0.6)
    plt.plot(pred[sample_idx], label='Prediction', color='red', alpha=0.6, linestyle='--')
    plt.title(f"Sample {sample_idx}: ICA Weights Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制 Correlation Scatter
    plt.subplot(1, 2, 2)
    plt.scatter(target.flatten(), pred.flatten(), alpha=0.1, s=3)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
    plt.xlabel("Ground Truth Value")
    plt.ylabel("Predicted Value")
    plt.title("Correlation: Truth vs Pred")
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些统计数据
    print(f"Pred Mean: {pred.mean():.4f}, Std: {pred.std():.4f}")
    print(f"Target Mean: {target.mean():.4f}, Std: {target.std():.4f}")

if __name__ == "__main__":
    visualize_prediction()