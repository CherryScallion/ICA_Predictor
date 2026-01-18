# 训练配置指南 (Training Configuration Guide)

## 🎯 快速开始

运行训练：
```bash
python HachimiNetV1/main.py
```

## 📋 GPU/CPU 设置

### 问题：为什么训练在CPU上？
你的系统没有检测到GPU。这可能是因为：
1. **没有安装NVIDIA GPU**
2. **没有安装CUDA** - 需要 CUDA 12.1+ 和 cuDNN 
3. **PyTorch未配置CUDA支持** - 需要安装 `torch[cuda]`

### 解决方案

**方案1：强制使用GPU（如果有硬件）**

编辑 `HachimiNetV1/configs/train_config.yaml`：
```yaml
training:
  device: "cuda"  # 改为 "cuda" 强制使用GPU
  # 或保持 "auto" 让系统自动选择
```

**方案2：检查和安装CUDA支持**

```bash
# 检查当前PyTorch是否支持CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果为False，重新安装支持CUDA的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚙️ 训练超参数调节

所有训练参数都在 `HachimiNetV1/configs/train_config.yaml` 中的 `training` 部分：

```yaml
training:
  # ===== GPU 设置 =====
  device: "auto"      # "auto" (自动), "cuda" (强制GPU), "cpu" (强制CPU)
  
  # ===== 训练基础参数 =====
  num_epochs: 50              # 训练轮数（越多越好，但时间长）
  batch_size: 4               # 批次大小（GPU内存充足可增加到8,16,32）
  learning_rate: 1.0e-4       # 学习率（通常在1e-3到1e-5之间）
  weight_decay: 1.0e-5        # L2正则化强度
  
  # ===== 优化器设置 =====
  optimizer: "AdamW"          # "AdamW" 或 "Adam"
  beta1: 0.9
  beta2: 0.999
  epsilon: 1.0e-8
  
  # ===== 梯度处理 =====
  gradient_clip_norm: 1.0     # 梯度裁剪阈值（防止梯度爆炸）
  
  # ===== Loss 函数权重 =====
  regression_loss_weight: 1.0       # MSE Loss权重
  cosine_loss_weight: 0.5           # Cosine相似度Loss权重
  
  # ===== 其他 =====
  num_workers: 0              # DataLoader进程数（0=主进程，>0=多进程）
  lazy_load: true             # 是否启用lazy loading（推荐true节省内存）
  checkpoint_interval: 5      # 每5个epoch保存一次模型
```

## 📊 常见调参建议

### 场景1：内存不足或显存不足
```yaml
batch_size: 2          # 减小批次大小
num_workers: 0         # 关闭多进程
lazy_load: true        # 启用lazy loading
```

### 场景2：想要更好的精度（需要更多时间和内存）
```yaml
num_epochs: 100        # 增加训练轮数
batch_size: 16         # 增大批次（如果显存足够）
learning_rate: 5.0e-5  # 降低学习率，训练更稳定
```

### 场景3：想要更快的训练速度
```yaml
num_epochs: 20         # 减少训练轮数
batch_size: 8          # 找平衡点
learning_rate: 1.0e-4  # 保持较高
num_workers: 4         # 如果CPU足够，增加DataLoader进程
```

### 场景4：GPU训练（推荐）
```yaml
device: "cuda"
batch_size: 16
learning_rate: 1.0e-4
num_workers: 4         # GPU时可以用多进程
```

## 🔧 代码中的其他可调参数

### main.py 中的参数

除了配置文件，还可以在 `main.py` 中直接修改：
```python
train_cfg = cfg.get('training', {})
num_epochs = train_cfg.get('num_epochs', 50)     # 默认值
batch_size = train_cfg.get('batch_size', 4)       # 默认值
lazy_load = train_cfg.get('lazy_load', True)      # 默认值
```

### models/loss.py 中的Loss权重

```python
# 回归Loss权重调节
loss_fn = WeightsRegressionLoss(
    lambda_cos=0.5  # Cosine损失的权重（0=仅MSE，1=平衡）
)
```

## 📈 训练监控

训练时会打印：
```
=====================================================================
📊 Training Config
=====================================================================
Epochs: 50
Batch Size: 4
Learning Rate: 0.0001
Lazy Load: True
=====================================================================

📌 Device: cuda:0
📌 CUDA Available: True
   GPU: NVIDIA RTX 3090
   VRAM: 24.0 GB

🚀 Training started on cuda:0
   Model Parameters: 123,456
   Trainable Parameters: 100,000
   
Epoch 1/50 | Train Loss: 0.5234 | Val Loss: 0.4892
Epoch 2/50 | Train Loss: 0.4856 | Val Loss: 0.4523
...
```

## 💾 Checkpoint保存

模型会每5个epoch（可配置）保存到 `./checkpoints/` 目录：
```
./checkpoints/
├── model_ep5.pth
├── model_ep10.pth
├── model_ep15.pth
└── ...
```

## 🐛 常见问题

**Q: 训练很慢？**
- A: 检查是否在CPU上训练。如果有GPU，确保 `device: "cuda"`

**Q: 显存不足（OOM）？**
- A: 减小 `batch_size` (4 → 2) 或启用 `lazy_load: true`

**Q: Loss不下降？**
- A: 尝试降低 `learning_rate` (1e-4 → 5e-5 或更低)

**Q: 想要断点续训？**
- A: 需要加载checkpoint，目前代码未实现，可自行添加

## 📝 完整配置示例

**GPU高性能训练：**
```yaml
training:
  device: "cuda"
  num_epochs: 100
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  optimizer: "AdamW"
  gradient_clip_norm: 1.0
  cosine_loss_weight: 0.5
  num_workers: 4
  lazy_load: true
  checkpoint_interval: 5
```

**CPU低配训练：**
```yaml
training:
  device: "cpu"
  num_epochs: 20
  batch_size: 2
  learning_rate: 5.0e-5
  weight_decay: 1.0e-5
  optimizer: "AdamW"
  gradient_clip_norm: 1.0
  cosine_loss_weight: 0.5
  num_workers: 0
  lazy_load: true
  checkpoint_interval: 10
```
