import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import DepthEstimationDataset2 as DepthEstimationDataset
from model_unet import UNet, Discriminator
from utils import visualize_sample, load_model_checkpoint, log_cosh_loss
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# 保存逐像素MSE损失图像为PNG文件，并使用均值和标准差进行归一化
def save_loss_map_as_image(loss_map, filename):
    # 计算均值和标准差用于归一化
    mean = loss_map.mean()
    std = loss_map.std()
    # 使用均值和标准差归一化损失图
    loss_map = (loss_map - mean) / std
    # 将值剪裁到[0, 1]之间用于可视化（根据数据分布可选）
    loss_map = torch.clamp(loss_map, 0, 1)
    # 转换为numpy数组并保存为图像
    loss_map_np = loss_map.cpu().numpy()[0, 0]  # 假设批量大小为1且深度通道为1
    plt.imsave(filename, loss_map_np, cmap='jet')

# 验证模型并保存逐像素的MSE损失图像
def validate_model(model, dataloader, criterion_depth, criterion_mask, device='cuda', output_dir='mse_images'):
    os.makedirs(output_dir, exist_ok=True)  # 创建保存MSE图像的目录

    model = model.to(device)
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (image, depth_gt, mask_gt) in enumerate(dataloader):
            image = image.to(device)
            depth_gt = depth_gt.to(device)
            mask_gt = mask_gt.to(device)

            output = model(image)
            depth_pred = output[:, 0:1, :, :]
            mask_pred = output[:, 1:2, :, :]

            # 计算深度预测的逐像素MSE损失（不进行归约）
            loss_depth_map = (depth_pred - depth_gt) ** 2

            # 将逐像素MSE损失保存为PNG图像
            save_loss_map_as_image(loss_depth_map, os.path.join(output_dir, f"mse_loss_{batch_idx}.png"))

            # 计算整体批次的减少损失
            loss_depth = criterion_depth(depth_pred, depth_gt)
            loss_mask = criterion_mask(mask_pred, mask_gt)

            loss = loss_depth * 0.9 + loss_mask * 0.1
            total_loss += loss.item() * image.size(0)
            num_samples += image.size(0)

            if batch_idx > 100:
                break

    avg_loss = total_loss / num_samples
    return avg_loss

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = DepthEstimationDataset(root_dir='datasets3/val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化模型
    generator_model = UNet(input_channels=1, output_channels=2, complexity=8, use_transformer=False)
    discriminator_model = Discriminator(complexity=4, input_channels=3)

    # 定义损失函数
    criterion_depth = nn.MSELoss(reduction='mean')  # 仍然使用均值归约来计算整体损失
    criterion_mask = nn.BCEWithLogitsLoss()

    # 获取最新的检查点文件
    checkpoint_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # 加载检查点
    generator_model, discriminator_model, epoch = load_model_checkpoint(generator_model, discriminator_model, checkpoint_path)
    print(f"从 {checkpoint_path} 加载检查点")

    # 验证模型并保存逐像素MSE图像
    avg_loss = validate_model(generator_model, dataloader, criterion_depth, criterion_mask, device)
    print(f"验证损失: {avg_loss:.8f}")
