import torch
import torch.nn as nn
import numpy as np

class HRFBlock(nn.Module):
    """
    [Module 2: The Physics Aligner]
    
    在当前阶段（Classification Pipeline），由于我们在 DataLoader 中已经手动对齐了
    EEG和fMRI的物理延迟 (delay=4s)，这个模块的作用转变为：
    '微调残留的相位误差' 和 '模拟神经到BOLD的低通滤波效应'。
    
    默认模式：Identity (不做处理，假设 DataLoader 对齐足够好)。
    进阶模式：可学习的 1D Conv 平滑层。
    """
    def __init__(self, n_components, learnable=False):
        super().__init__()
        self.learnable = learnable
        
        if self.learnable:
            # 初始化为一个简单的平滑卷积核 [1, 1, 3]
            # 允许模型微调 temporal smoothing
            self.smoothing = nn.Conv1d(
                in_channels=n_components, 
                out_channels=n_components,
                kernel_size=3,
                padding=1,
                groups=n_components # Depthwise conv，每个component独立平滑
            )
            # 初始化权重为 [0.2, 0.6, 0.2] 的高斯模糊
            nn.init.constant_(self.smoothing.weight, 0.2)
            self.smoothing.weight.data[:, 0, 1] = 0.6
        
    def forward(self, z_neural):
        """
        Input: [Batch, K] (如果是单帧模式)
               [Batch, Time, K] (如果是序列模式)
        """
        if not self.learnable:
            return z_neural
            
        # 如果是单帧输入 [B, K]，unsqueeze 时间维度进行假的卷积，再squeeze回来
        if z_neural.dim() == 2:
            return z_neural # 单帧无法做时序平滑
            
        # [B, T, K] -> [B, K, T]
        x = z_neural.permute(0, 2, 1)
        x = self.smoothing(x)
        # Back to [B, T, K]
        return x.permute(0, 2, 1)

def spm_hrf_basis(tr, duration=32.0):
    """
    (Optional Tool) 
    生成标准的 SPM HRF kernel (双 Gamma 函数)。
    如果后续想做完全物理可微的HRF，可以用这个初始化 Conv1d权重。
    """
    dt = tr
    t = np.arange(0, duration, dt)
    
    # SPM default parameters
    p = [6, 16, 1, 1, 6, 0] 
    
    # Double Gamma Function
    # h = (t/d1)^a1 * exp(-(t-d1)/b1) - c * (t/d2)^a2 * exp(-(t-d2)/b2)
    # (简化版实现)
    from scipy.stats import gamma
    
    hrf = gamma.pdf(t, p[0]/p[2], scale=dt*p[2]) - \
          p[5] * gamma.pdf(t, p[1]/p[3], scale=dt*p[3])
          
    hrf = hrf / np.sum(hrf)
    return torch.tensor(hrf).float()