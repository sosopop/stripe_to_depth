import torch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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
        image = torch.from_numpy(image).unsqueeze(0).float()  # 转换为 PyTorch Tensor，并加上通道维度

        # 读取深度图
        z_data = pd.read_csv(os.path.join(data_dir, 'z.csv'), header=None)
        depth_image = torch.from_numpy(z_data.values).unsqueeze(0).float()  # 加上通道维度

        # 读取掩码图
        mask_data = pd.read_csv(os.path.join(data_dir, 'mask.csv'), header=None)
        mask_image = torch.from_numpy(mask_data.values).unsqueeze(0).float()

        return image, depth_image, mask_image

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
