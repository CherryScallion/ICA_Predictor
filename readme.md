Physics-E2fNet: 操作手册
Physics-E2fNet 是一个受物理启发的 EEG 到 fMRI 生成模型。它摒弃了传统的“体素级回归”范式，转而采用 "神经状态分类 (Neural State Classification)" + "基底重构 (Basis Reconstruction)" 的思路，以解决小样本数据下的泛化差、边缘模糊和均值坍缩问题。
0. 环境依赖
code
Bash
pip install torch numpy nibabel nilearn scipy scikit-learn tqdm pyyaml
1. 配置文件 (configs/train_config.yaml)
一切的起点。你需要修改文件中的路径，使其指向你的 BIDS 或 .nii 数据。
raw_fmri_dir: 指向 fMRI .nii.gz 文件所在的顶层文件夹。
raw_eeg_dir: 指向 EEG 数据的文件夹（注意脚本 loaders.py 需要你保证 EEG 和 fMRI 有文件名对应关系，如 sub01_rest.nii 对应 sub01_rest_eeg.pt）。
data_specs:
n_ica_components: 默认为 64。建议不要超过 128，否则会OOM。
activation_threshold: 判定激活的 Z-score 阈值，默认为 1.0 (约前 15% 的高亮区域)。
2. 数据处理流水线 (The Pipeline)
本项目严禁在训练时进行在线处理，必须运行以下离线处理脚本：
第一步：空间基底提取 (Group ICA)
脚本: python processing/run_ica.py
作用:
遍历所有训练数据，训练 Group ICA 模型。
在 data/templates/ 下生成 ica_basis.pt (你的解剖学字典) 和 gray_mask.pt (灰质约束)。
耗时: 取决于数据量，约 10-30 分钟。只需运行一次。
第二步：分类标签生成 (Label Generation)
脚本: python processing/quantize_data.py
作用:
读取 NIfTI 文件，Mask 掉非脑区。
执行 Run-wise Z-score 标准化。
离散化为三分类标签: {0:背景, 1:激活, 2:抑制, -1:忽略}。
在 data/processed/ 下生成轻量级的 _labels.pt 文件。
注意: 此处你还需确保你的 EEG 数据也被处理成 PyTorch Tensor 并放在相应位置。
3. 模型训练 (Training)
脚本: python main.py
流程:
加载阶段: 模型会自动载入 ica_basis.pt 到 GPU 内存中，将其作为固定的“解码器”权重。
Warm-up 阶段 (Epoch 0-5):
Loss 仅包含 Focal Loss。
目的是让模型先学会分清“哪是背景，哪是脑子”，避免一开始就优化几何形状导致 Loss 震荡。
Refine 阶段 (Epoch 5+):
Dice Loss 权重逐渐从 0 加到 1.0。
强迫模型预测的激活区域在空间上是成块的、连贯的，消除椒盐噪声。
4. 关键问答 (Q&A)
Q: 为什么生成的图像没有锐利的边缘？
A: 本模型的原理是 ICA 基底叠加。如果你用的是 CanICA，基底本身是相对平滑的（Gaussian blob-like）。如果你想要更锐利，请在 run_ica.py 中调小 smoothing_fwhm 参数，或者换用 DictLearning 方法。
Q: 显存爆了 (OOM) 怎么办？
A: fMRI 是 3D Volume，显存占用极大。
在 train_config.yaml 降低分辨率 (如 [53, 63, 52] 实际上是 4mm 分辨率)。
降低 batch_size (建议 batch=4 或 8)。
使用 16-bit 混合精度训练 (需要简单修改 Trainer)。
Q: 模型一直在预测全黑 (全背景)？
A:
检查 activation_threshold 是否设太高 (例如 3.0)，导致标签里全是背景。
增加 Focal Loss 的 gamma 值。
检查是否关掉了 Dice Loss 的 Warm-up。Dice Loss 在初期非常不稳定。