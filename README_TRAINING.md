# HachimiNetV1 训练快速参考

## 📌 常见问题解答

### Q1: 为什么训练在CPU上，不在GPU上？
**原因：** 你的系统没有可用的GPU，或PyTorch未配置CUDA支持

**解决方案：**
1. **检查GPU**: 运行 `python -c "import torch; print(torch.cuda.is_available())"`
2. **如果是False**，需要安装CUDA支持的PyTorch：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. **或修改配置强制CPU** (如果没有GPU)：
   编辑 `HachimiNetV1/configs/train_config.yaml`
   ```yaml
   training:
     device: "cpu"  # 改为cpu
   ```

---

## ⚙️ 训练参数在哪里调节？

### 主配置文件：`HachimiNetV1/configs/train_config.yaml`

**GPU/Device设置：**
```yaml
training:
  device: "auto"  # "auto"(自动), "cuda"(强制GPU), "cpu"(强制CPU)
```

**关键训练参数：**
```yaml
training:
  num_epochs: 50                  # 训练轮数
  batch_size: 4                   # 批大小 (↑内存使用, ↓收敛稳定性)
  learning_rate: 1.0e-4           # 学习率 (↑收敛快但可能振荡, ↓收敛慢但稳定)
  weight_decay: 1.0e-5            # L2正则化
  optimizer: "AdamW"              # 优化器类型
  gradient_clip_norm: 1.0         # 梯度裁剪
```

**Loss函数权重：**
```yaml
  cosine_loss_weight: 0.5         # Cosine相似度Loss的权重
```

**其他参数：**
```yaml
  num_workers: 0                  # DataLoader线程数 (CPU足够时可改>0)
  lazy_load: true                 # 是否延迟加载数据 (推荐true)
  checkpoint_interval: 5          # 每5个epoch保存一次模型
```

---

## 🚀 快速调参方案

### 方案A：我有GPU (推荐)
```yaml
training:
  device: "cuda"          # 使用GPU
  batch_size: 16          # 增大批大小 (GPU可以处理)
  num_epochs: 100         # 更多轮数以获得更好效果
  num_workers: 4          # 多进程加载数据
  learning_rate: 1.0e-4
```

### 方案B：没有GPU，CPU训练
```yaml
training:
  device: "cpu"           # 强制使用CPU
  batch_size: 2           # 减小批大小节省内存
  num_epochs: 20          # 缩减轮数加快训练
  num_workers: 0          # 关闭多进程
  lazy_load: true         # 启用延迟加载
  learning_rate: 5.0e-5   # 降低学习率更稳定
```

### 方案C：内存不足 (OOM)
```yaml
training:
  batch_size: 2           # 减小到2或1
  num_workers: 0
  lazy_load: true
  learning_rate: 1.0e-4
```

### 方案D：想要更好的精度 (时间充足)
```yaml
training:
  num_epochs: 200         # 加倍
  batch_size: 32          # 增大批大小 (需要足够显存)
  learning_rate: 5.0e-5   # 降低学习率
  num_workers: 4
```

---

## 📊 训练输出说明

运行 `python HachimiNetV1/main.py` 时，你会看到：

```
60============================================================
📊 Training Config
============================================================
Epochs: 50
Batch Size: 4
Learning Rate: 0.0001
Lazy Load: True
============================================================

📌 Device: cuda:0
📌 CUDA Available: True
   GPU: NVIDIA RTX 3090
   VRAM: 24.0 GB

🚀 Training started on cuda:0
   Model Parameters: 123,456
   Trainable Parameters: 100,000

Train Ep 1:  10%|████                  | 102/1000 [00:15<02:30, 5.98it/s]
Train Ep 1: 100%|██████████████████    | 1000/1000 [02:30<00:00, 6.67it/s, loss=0.524, mse=0.48, cos=0.19]
Epoch 1/50 | Train Loss: 0.5234 | Val Loss: 0.4892

Train Ep 2:  ...
Epoch 2/50 | Train Loss: 0.4856 | Val Loss: 0.4523

...
```

**关键指标：**
- `loss`: 总损失值（应该逐渐下降）
- `mse`: 回归损失（预测权重的均方误差）
- `cos`: Cosine相似度损失

---

## 💾 模型保存位置

训练过程中的模型会保存到：
```
./checkpoints/
├── model_ep5.pth
├── model_ep10.pth
├── model_ep15.pth
└── model_ep50.pth
```

修改间隔：
```yaml
training:
  checkpoint_interval: 5  # 改为其他数字
```

---

## 🔧 代码中的其他参数

### main.py 中可以修改的

```python
# 数据加载
train_loader = DataLoader(
    dataset, 
    batch_size=batch_size,      # 从config读取
    shuffle=True, 
    num_workers=num_workers,    # 从config读取
    pin_memory=(device.type == 'cuda')
)

# Loss函数
loss_fn = WeightsRegressionLoss(
    lambda_cos=train_cfg.get('cosine_loss_weight', 0.5)  # 从config读取
)
```

### training/trainer.py 中的学习率调度

目前没有学习率衰减。如果想添加，可以在 `_update_scheduler` 方法中实现。

---

## 📈 监控训练进度

**检查损失是否下降：**
- Loss应该从高值逐渐降低
- 如果loss不动或增加，说明学习率可能太高

**检查过拟合：**
- 如果 Val Loss 持续增加而 Train Loss 继续下降，说明过拟合
- 可以增加 weight_decay 或减少 num_epochs

**检查GPU使用：**
```bash
# 在另一个终端运行
nvidia-smi
```

---

## 🆘 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|-------|--------|
| 训练很慢 | CPU训练 | 检查device设置，安装CUDA |
| OOM错误 | 显存不足 | 减小batch_size |
| Loss不下降 | 学习率太高 | 降低learning_rate (10倍) |
| Loss=NaN | 梯度爆炸 | 降低learning_rate或增加gradient_clip_norm |
| 精度不好 | 训练不足 | 增加num_epochs或减小learning_rate |

---

## 📝 完整配置示例

**最小化配置（快速测试）：**
```yaml
training:
  device: "auto"
  num_epochs: 5
  batch_size: 2
  learning_rate: 1.0e-4
```

**完整高效配置：**
```yaml
training:
  device: "auto"
  num_epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  optimizer: "AdamW"
  beta1: 0.9
  beta2: 0.999
  epsilon: 1.0e-8
  gradient_clip_norm: 1.0
  regression_loss_weight: 1.0
  cosine_loss_weight: 0.5
  num_workers: 4
  lazy_load: true
  checkpoint_interval: 5
```

---

## 更多信息

详细的训练配置指南见：`TRAINING_CONFIG_GUIDE.md`
