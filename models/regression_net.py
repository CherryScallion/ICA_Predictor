import torch
import torch.nn as nn

class HachimiRegressionNet(nn.Module):
    """
    输入: [Batch, 20, 64, 249] (EEG Feature Maps)
    输出: [Batch, 64] (ICA Component Weights)
    """
    def __init__(self, in_channels=20, n_out=64):
        super().__init__()
        
        # 针对 64x249 的长方形特征图设计 CNN
        self.features = nn.Sequential(
            # Layer 1: [B, 32, 32, 125] (stride 2)
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            # Layer 2: [B, 64, 16, 63]
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            # Layer 3: [B, 128, 8, 32]
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            # Layer 4: [B, 256, 4, 16]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        
        # 全局池化 -> [B, 256, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, n_out) # Output 64 weights
        )

    def forward(self, x):
        # 期望 x shape: [B, 20, 64, 249]
        x = self.features(x)
        x = self.pool(x)
        out = self.regressor(x)
        return out