# 深度图估计项目

## 项目概述

本项目实现了一个基于深度学习的单目深度估计系统，能够从单张RGB图像中预测深度图和有效区域掩码。该系统采用结合U-Net和Transformer的混合架构，并引入GAN进行半监督学习，有效解决了深度估计中的数据稀缺问题。

![demo](https://github.com/sosopop/stripe_to_depth/raw/master/assets/demo1.jpg) 
![demo](https://github.com/sosopop/stripe_to_depth/raw/master/assets/demo2.jpg) 

## 项目特点

- **混合网络架构**：结合U-Net的高分辨率特征提取能力和Transformer的全局上下文建模能力
- **双输出设计**：同时输出深度图和有效区域掩码，提高深度预测的可靠性
- **半监督学习**：采用GAN架构进行半监督训练，充分利用无标签数据
- **多尺度特征融合**：通过跳跃连接实现多层次特征融合，提高边缘细节保留
- **灵活复杂度调整**：提供不同复杂度模型（低、中、高），适应不同硬件环境
- **残差连接判别器**：设计了具有残差连接的判别器，提高GAN训练稳定性

## 网络架构

### 生成器 (U-Net + Transformer)

生成器采用改进的U-Net结构，包括：

- 7层编码器网络，逐步提取高级特征
- 中间Transformer层，建模全局上下文依赖
- 7层解码器网络，通过跳跃连接恢复空间细节
- 双通道输出：深度图和掩码图

![unet](https://github.com/sosopop/stripe_to_depth/raw/master/assets/unet.png) 

### 判别器

判别器采用改进的卷积网络结构：

- 交替使用3×3和5×5卷积核，增强感受野多样性
- 残差连接结构，提高训练稳定性
- 池化层和BatchNorm层，加速收敛
- 最终通过全连接层输出真假判别结果

![discriminator](https://github.com/sosopop/stripe_to_depth/raw/master/assets/discriminator.png) 

## 训练策略

本项目采用创新的两阶段训练策略：

1. **有监督预训练阶段**：
   - 使用带标签数据进行基础训练
   - 联合深度损失和掩码损失优化网络参数

2. **半监督GAN微调阶段**：
   - 使用预训练的生成器产生无标签数据的伪标签
   - 判别器学习区分真实和生成的深度图
   - 生成器通过对抗学习进一步优化深度估计

![架构流程图](https://github.com/sosopop/stripe_to_depth/raw/master/assets/architecture.png) 

## 数据增强技术

- 随机水平翻转
- 随机擦除（模拟遮挡情况）
- 添加噪声（提高模型鲁棒性）

## 损失函数设计

- **深度估计损失**：MSE损失 / Log-Cosh损失
- **掩码预测损失**：BCE损失
- **判别器损失**：标准GAN损失

## 使用方法

### 环境配置

```bash
pip install torch torchvision tqdm tensorboard
```

### 训练模型

```bash
python train.py
```

### 参数说明

- `datasets_dir`: 数据集路径
- `use_data_enhance`: 是否使用数据增强
- `complexity`: 模型复杂度 (8/16/32)
- `use_transformer`: 是否使用Transformer模块

## 可视化结果

训练过程中，系统会自动保存以下可视化内容：

- 输入RGB图像
- 预测深度图
- 预测掩码图
- 真实深度图和掩码图（如果有）
- TensorBoard训练指标

## 未来工作

- 引入注意力机制进一步提高细节预测
- 探索更高效的网络结构以减少参数量
- 添加时序信息以提高视频序列深度估计质量
- 扩展到更多领域应用场景