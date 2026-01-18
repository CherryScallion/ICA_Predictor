import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisProjector(nn.Module):
    """
    [Module 3 & 4: Basis Projection & Semantic Segmentation]
    
    Pipeline:
    1. Linear Combination: Weighted Sum(Basis, z_bold) -> 粗糙强度体 [B, 1, D, H, W]
    2. Segmentation Head: 1x1 Conv3d -> 映射到类别Logits [B, 3, D, H, W]
    """
    def __init__(self, basis_buffer_name='spatial_basis'):
        super().__init__()
        self.basis_buffer_name = basis_buffer_name
        
        # [Module 4] 语义分割头
        # 为什么是 1x1 Conv? 
        # 因为我们不需要改变空间结构，只需要根据当前点的强度
        # 学习一个非线性映射，判断它是 0(背景), 1(激活), 还是 2(抑制)
        self.classifier = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=1), # 升维增加非线性特征容量
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=1)  # 输出 3 个类别 Logits
        )

    def forward(self, weights, model_buffer):
        """
        weights: [Batch, K] (神经激活强度)
        model_buffer: 这是一个容器 (self of main model)，用于获取注册的 buffer
        """
        # 1. 获取预注册的 ICA 基底
        # Shape: [K, D, H, W]
        if not hasattr(model_buffer, self.basis_buffer_name):
            raise RuntimeError(f"Buffer {self.basis_buffer_name} not found in model!")
            
        basis_maps = getattr(model_buffer, self.basis_buffer_name)
        
        # 2. 线性投影 (Vector to Volume)
        # Einsum: 对于每个batch b, 对 K 个维度进行加权求和
        # w[b, k] * map[k, d, h, w] -> vol[b, d, h, w]
        rough_volume = torch.einsum('bk, kdhw -> bdhw', weights, basis_maps)
        
        # 增加 Channel 维度供 Conv3d 使用
        # [B, D, H, W] -> [B, 1, D, H, W]
        x = rough_volume.unsqueeze(1)
        
        # 3. 语义分割分类
        # Logits: [B, 3, D, H, W]
        logits = self.classifier(x)
        
        return logits