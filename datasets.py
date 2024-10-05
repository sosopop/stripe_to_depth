import torch
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tifffile as tiff  # 用于读取 tiff 文件
import random
from torchvision import transforms
import numpy as np

class DepthEstimationDataset2(Dataset):
    def __init__(self, root_dir, data_size=0, transform=None, noise_mask_prob=(0.0, 0.00), z_range=(-0.4, -0.3)):
        self.root_dir = root_dir
        self.data_size = data_size
        self.noise_mask_prob = noise_mask_prob
        self.z_range = z_range
        # 获取所有以 _image.png 结尾的文件，作为数据集的标识
        self.data_list = [f[:-10] for f in os.listdir(root_dir) if f.endswith('_image.png')]
        self.transform = transform

    def __len__(self):
        if self.data_size != 0:
            return self.data_size
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.data_size != 0:
            idx = random.randint(0, len(self.data_list)-1)  # 随机选择一个数据
        file_base = self.data_list[idx]
        
        # 构建文件路径
        image_path = os.path.join(self.root_dir, f'{file_base}_image.png')
        mask_path = os.path.join(self.root_dir, f'{file_base}_mask.png')
        depth_path = os.path.join(self.root_dir, f'{file_base}_z.tiff')

        # 读取条纹图像 (image)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0  # 转为 PyTorch Tensor，并添加通道维度

        # 读取掩码图 (mask)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_image = torch.from_numpy(mask_image).unsqueeze(0).float()
        mask_image[mask_image > 0] = 1.0
        
        # 读取深度图 (z.tiff)
        depth_image = tiff.imread(depth_path)
        depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()  # 转为 PyTorch Tensor，并添加通道维度

        # 计算随机偏移量 (可以根据需要启用随机偏移)
        # offset_y, offset_x = 0, 0  # 此处可以更改为调用 self.get_random_offset 来随机偏移
        offset_y, offset_x = self.get_random_offset(image, 576, 576)

        # 使用相同的偏移量填充 image、depth_image 和 mask_image
        image = self.pad_to_size_with_offset(image, 576, 576, offset_y, offset_x)
        depth_image = self.pad_to_size_with_offset(depth_image, 576, 576, offset_y, offset_x)
        mask_image = self.pad_to_size_with_offset(mask_image, 576, 576, offset_y, offset_x)

        # 如果设置了数据增强，则应用于 image 和 mask_image (注意：不对 depth_image 做增强)
        if self.transform:
            # 合并 image 和 mask_image 进行同步变换
            stack = torch.cat([image, depth_image, mask_image], dim=0)
            stack = self.transform(stack)
            image, depth_image, mask_image = stack[0:1, :, :], stack[1:2, :, :], stack[2:3, :, :]  # 分离增强后的 image 和 mask_image
        
        noise_mask = self.make_noise_mask(p=self.noise_mask_prob)
        image = image * (1 - noise_mask)
        # mask_image = mask_image * (1 - noise_mask)
        # depth_image = depth_image * (1 - noise_mask)
        
        depth_image[mask_image == 0] = -1
        
        depth_image = depth_image.clamp(self.z_range[0], self.z_range[1])
        depth_image = (depth_image - self.z_range[0]) / (self.z_range[1] - self.z_range[0])

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

    def make_noise_mask(self, p=(0.0, 0.03), size=(576, 576)):
        """
        创建一个随机噪声掩膜。

        参数：
        - p: 添加噪声的概率 (float)，范围在 [0, 1]。
        - noise_scale: 噪声的大小范围 (tuple)，表示噪声掩膜的比例。
        - size: 噪声掩膜的大小 (tuple)，例如 (高, 宽)。

        返回：
        - noise_mask: 随机生成的噪声掩膜，值为 0 或 1。
        """
        p = random.uniform(p[0], p[1])  # 随机选择噪声的概率
        # 生成一个随机的掩膜，值为0到1之间的浮点数
        random_mask = torch.rand(size)

        # 创建噪声掩膜，根据概率p生成0和1
        noise_mask = (random_mask < p).float()  # 转换为float型

        return noise_mask

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transforms.RandomErasing(p=0.8, scale=(0.07, 0.07), ratio=(0.5, 2), value=0),
        transforms.RandomErasing(p=0.8, scale=(0.05, 0.05), ratio=(0.5, 2), value=0),
        transforms.RandomErasing(p=0.8, scale=(0.1, 0.1), ratio=(0.5, 2), value=0)
    ])
    # 创建数据集和数据加载器
    dataset = DepthEstimationDataset2(root_dir='datasets3/val/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 可视化示例
    for idx, (image, depth_image, mask_image) in enumerate(dataloader):
        if idx >= 5:  # 只显示前5条数据
            break
        image = image.squeeze(0).squeeze(0)
        depth_image = depth_image.squeeze(0).squeeze(0)
        mask_image = mask_image.squeeze(0).squeeze(0)

        # 掩码处理后的深度图归一化
        masked_z = torch.where(mask_image > 0, depth_image, torch.nan)
        z_mask_data = masked_z[~torch.isnan(masked_z)]
        
        z_mask_normalized_data = masked_z
        if len(z_mask_data) != 0:
            z_mask_max = torch.max(z_mask_data)
            z_mask_min = torch.min(z_mask_data)
            z_mask_normalized_data = (masked_z - z_mask_min) / (z_mask_max - z_mask_min)
            z_mask_normalized_data[torch.isnan(z_mask_normalized_data)] = 0
        
        masked_z = depth_image

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

        # 掩码后的深度图
        axes[2].imshow(masked_z, cmap='jet')
        axes[2].set_title('Masked Depth Image')
        axes[2].axis('off')
        
        # 掩码后的归一化深度图
        axes[3].imshow(z_mask_normalized_data, cmap='jet')
        axes[3].set_title('Masked & Normalized Depth Image')
        axes[3].axis('off')

        # 展示图像
        plt.tight_layout()
        plt.show()
