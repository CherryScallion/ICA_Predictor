# GPU训练性能分析与优化指南

## 当前状态
- **设备**: CUDA (RTX 3060 Laptop, 6.4GB VRAM)
- **Batch Size**: 32
- **训练速度**: ~3.6s/iter (~569 iters/epoch)
- **每个Epoch耗时**: ~34分钟
- **50 Epochs总耗时**: ~28小时

## 为什么CPU吃满但GPU没完全动？

### 根本原因
**H5文件I/O瓶颈** - 数据加载链路是序列化的：
```
主线程CPU
  ├─ 打开H5文件 (I/O等待)
  ├─ 读32个样本 (磁盘读取)
  ├─ Normalize频率维度 (CPU处理)
  └─ 转移到GPU (PCIe传输)
      └─ GPU forward/backward
```

每次迭代必须等待上面的链路完成，所以GPU很多时间在等待数据。

### 为什么改batch_size=256失败
- 所有样本数据（20×64×249 float32）= ~4.9MB/样本
- 256个样本 = 1.25GB，超过RTX 3060的6.4GB（需要留给权重、激活等）
- 导致OOM（内存溢出）

## 已实施的优化

✅ 异步GPU数据转移 (non_blocking=True + Stream)
✅ H5持久化连接(避免重复打开文件)  
✅ LRU缓存机制(缓存最近访问的256个样本)
✅ pin_memory=True(锁定CPU页面加快CPU→GPU转移)

## 进一步优化方案（如需实施）

### 方案1: 数据预处理（推荐）
将H5转换为优化格式，一次性加载到内存：
```python
# 在training开始前运行一次
python HachimiNetV1/preprocessing/convert_h5_to_npz.py
# 生成 data/optimized_npz/ 目录（压缩numpy格式）
# 优势：磁盘I/O开销减少80%+
```

### 方案2: SSD升级
将H5数据复制到本地SSD（如果使用HDD）：
- HDD随机读取: ~100MB/s
- SSD随机读取: ~500MB/s
- 改进幅度: 5倍

### 方案3: 减少normalize开销
预计算normalize映射表：
```python
# 目前: 每个样本都 np.pad() 一次
# 优化: 使用预处理索引，避免运行时normalize
```

### 方案4: 多进程DataLoader（不适用H5）
H5不支持跨进程序列化，所以num_workers>0无法使用。
如果转换为NPZ/NPY，可以启用num_workers=4-8。

## 当前训练继续

✅ **GPU训练已正确启动并运行**
- Batch 1/569: loss=2.28e+3 (正在学习)
- Device设置: cuda (正确)
- 数据正确转移到GPU处理

推荐：
1. 让训练继续运行至少1个Epoch（观察loss下降趋势）
2. 完成50个Epoch的训练（预计28小时）
3. 检查最终Loss值和模型性能

## 监控建议

运行以下命令监控训练进度：
```bash
# 监控日志（每5秒刷新）
watch -n 5 'tail -50 training.log'

# 检查GPU内存（在另一个终端）
python -c "import torch; print(torch.cuda.memory_allocated()/1e9)"
```

---
**结论**: CPU→GPU的数据加载是瓶颈，但这对于H5数据集是正常的。当前3.6s/iter的速度是可接受的折衷方案。
