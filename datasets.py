import torch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

class DepthEstimationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dir = os.path.join(self.root_dir, self.data_list[idx])

        # 读取图像
        image_path = os.path.join(data_dir, '1.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0  # 转换为 PyTorch Tensor，并加上通道维度

        # 读取深度图
        z_data = pd.read_csv(os.path.join(data_dir, 'z.csv'), header=None)
        depth_image = torch.from_numpy(z_data.values).unsqueeze(0).float()  # 加上通道维度
        
        # depth_gt 归一化到 -0.4~-0.3 之间
        # depth_image = depth_image.clamp(-0.38, -0.3)
        # depth_image = (depth_image + 0.4) / 0.1

        # 读取掩码图
        mask_data = pd.read_csv(os.path.join(data_dir, 'mask.csv'), header=None)
        mask_image = torch.from_numpy(mask_data.values).unsqueeze(0).float()

        # 计算随机偏移量
        # offset_y, offset_x = self.get_random_offset(image, 576, 576)
        offset_y, offset_x = 0, 0

        # 使用相同的偏移量填充 image、depth_image 和 mask_image
        image = self.pad_to_size_with_offset(image, 576, 576, offset_y, offset_x)
        depth_image = self.pad_to_size_with_offset(depth_image, 576, 576, offset_y, offset_x)
        mask_image = self.pad_to_size_with_offset(mask_image, 576, 576, offset_y, offset_x)

        return image, depth_image, mask_image

    def get_random_offset(self, tensor, target_height, target_width):
        """
        生成随机偏移量，确保不会溢出边界。
        """
        _, height, width = tensor.shape
        pad_bottom = target_height - height
        pad_right = target_width - width

        # 生成随机偏移量，保证不会溢出
        offset_y = random.randint(0, pad_bottom)
        offset_x = random.randint(0, pad_right)

        return offset_y, offset_x

    def pad_to_size_with_offset(self, tensor, target_height, target_width, offset_y, offset_x):
        """
        将输入的 tensor 填充到目标尺寸 target_height x target_width，
        并使用指定的偏移量，将图像随机偏移到不同位置，且不溢出。
        """
        _, height, width = tensor.shape
        pad_bottom = target_height - height
        pad_right = target_width - width

        # 使用传入的偏移量来确定填充区域
        padding = (offset_x, pad_right - offset_x, offset_y, pad_bottom - offset_y)  # (left, right, top, bottom)
        return F.pad(tensor, padding, "constant", 0)  # 用0填充


if __name__ == '__main__':
    # 创建数据集和数据加载器
    dataset = DepthEstimationDataset(root_dir='data')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 可视化示例
    for image, depth_image, mask_image in dataloader:
        image = image.squeeze(0).squeeze(0)
        depth_image = depth_image.squeeze(0).squeeze(0)
        mask_image = mask_image.squeeze(0).squeeze(0)

        # 深度图归一化处理
        z_max = torch.max(depth_image)
        z_min = torch.min(depth_image)
        z_normalized_data = (depth_image - z_min) / (z_max - z_min)

        # 掩码处理后的深度图归一化
        masked_z = torch.where(mask_image == 1, depth_image, torch.nan)
        z_mask_data = masked_z[~torch.isnan(masked_z)]
        z_mask_max = torch.max(z_mask_data)
        z_mask_min = torch.min(z_mask_data)
        z_mask_normalized_data = (masked_z - z_mask_min) / (z_mask_max - z_mask_min)
        z_mask_normalized_data[torch.isnan(z_mask_normalized_data)] = 0

        # 使用matplotlib显示
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 掩码图
        axes[1].imshow(mask_image, cmap='gray')
        axes[1].set_title('Mask Image')
        axes[1].axis('off')

        # 归一化深度图
        axes[2].imshow(z_normalized_data, cmap='jet')
        axes[2].set_title('Normalized Depth Image')
        axes[2].axis('off')

        # 掩码后的归一化深度图
        axes[3].imshow(z_mask_normalized_data, cmap='jet')
        axes[3].set_title('Masked & Normalized Depth Image')
        axes[3].axis('off')

        # 展示图像
        plt.tight_layout()
        plt.show()

        break  # 只展示一个批次
