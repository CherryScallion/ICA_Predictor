# models/loss.py
import torch
import torch
import torch.nn as nn

class WeightsRegressionLoss(nn.Module):
    def __init__(self, lambda_cos=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        # 沿着维度1 (Component维度) 计算相似度
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.lambda_cos = lambda_cos

    def forward(self, pred, target):
        """
        pred:   [Batch, 64]
        target: [Batch, 64]
        """
        # 1. 基础回归：数值要准
        loss_mse = self.mse(pred, target)
        
        # 2. 模式匹配：方向要对 (最大化相似度 -> 最小化 1-Sim)
        # CosineSim 返回 [-1, 1], 越接近 1 越好
        cos_sim = self.cosine(pred, target).mean() 
        loss_cos = 1.0 - cos_sim
        
        # 组合
        total_loss = loss_mse + self.lambda_cos * loss_cos
        
        return total_loss, {"mse": loss_mse.item(), "cos": loss_cos.item()}