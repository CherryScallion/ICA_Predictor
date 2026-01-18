# models/classifier_net.py
import torch
import torch.nn as nn
from models.components.temporal import EEGEncoder
from models.components.physics import HRFBlock        # 新增
from models.components.projector import BasisProjector # 新增
from utils.paths import get_template_dir, resolve_path

class PhysicsE2fNet(nn.Module):
    def __init__(self, 
                 n_ica_components=64,
                 eeg_channels=64,
                 eeg_time_len=300,
                 basis_path=None,
                 task='regression'):
        super().__init__()
        
        self.task = task  # 'regression' or 'segmentation'
        
        # 解析 basis_path，如果未提供则使用默认路径
        if basis_path is None:
            basis_path = get_template_dir() / 'ica_mixing_matrix.pt'
        else:
            basis_path = resolve_path(basis_path)
        
        # 1. 神经驱动器 [B, C, F, T] -> [B, K]
        self.driver = EEGEncoder(
            in_channels=eeg_channels,
            n_components=n_ica_components
        )
        
        # 2. 物理过滤器 (Optional Phase Correction / Smoothing)
        self.physics = HRFBlock(n_components=n_ica_components, learnable=False)
        
        if task == 'regression':
            # For weight regression, just return [B, K] directly
            pass
        else:
            # 3. 静态 ICA 基底加载 (for segmentation)
            try:
                basis_tensor = torch.load(str(basis_path)) # [K, D, H, W]
                print(f"Loaded Basis: {basis_tensor.shape}")
            except Exception as e:
                print(f"⚠️ Warning: Basis not found at {basis_path}. Using Random.")
                basis_tensor = torch.randn(n_ica_components, 53, 63, 52)
            
            self.register_buffer('spatial_basis', basis_tensor)
            
            # 4. 投影与分割头 (Component Based)
            self.projector = BasisProjector(basis_buffer_name='spatial_basis')
        
    def forward(self, eeg_segment):
        """
        Args:
            eeg_segment: [B, C, F, T] (channels, frequency bins, time)
        
        Returns:
            - For regression: [B, K] predicted ICA weights
            - For segmentation: [B, 3, D, H, W] logits
        """
        # Step 1: Encoder -> Latent Weights [B, K]
        z_neural = self.driver(eeg_segment)
        
        # Step 2: Physics Filter (HRF Smoothing) -> [B, K]
        z_bold = self.physics(z_neural)
        
        if self.task == 'regression':
            # Direct weight prediction
            return z_bold
        else:
            # Step 3: Projection + Segmentation Head -> Logits
            logits = self.projector(z_bold, self)
            return logits