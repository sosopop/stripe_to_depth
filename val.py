import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets2 import DepthEstimationDataset2 as DepthEstimationDataset
from model_unet import UNet, Discriminator
from utils import visualize_sample, load_model_checkpoint
import os

def validate_model(model, dataloader, criterion_depth, criterion_mask, device='cuda'):
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

            depth_mask_gt = depth_gt * mask_gt
            depth_mask_pred = depth_pred * torch.sigmoid(mask_pred)

            loss_depth = criterion_depth(depth_mask_gt, depth_mask_pred)
            loss_mask = criterion_mask(mask_pred, mask_gt)

            loss = loss_depth * 0.9 + loss_mask * 0.1
            total_loss += loss.item() * image.size(0)
            num_samples += image.size(0)

            for i in range(image.size(0)):
                visualize_sample(
                    image[i].cpu(),
                    depth_pred[i].cpu(),
                    mask_pred[i].cpu(),
                    depth_gt[i].cpu(),
                    mask_gt[i].cpu(),
                    f"validation_batch{batch_idx}_sample{i}"
                )
            
            break

    avg_loss = total_loss / num_samples
    return avg_loss

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = DepthEstimationDataset(root_dir='datasets/val')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # 初始化模型
    generator_model = UNet(input_channels=1, output_channels=2, complexity=8)
    discriminator_model = Discriminator(complexity=4)

    # 定义损失函数
    criterion_depth = nn.MSELoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    # 获取最新的checkpoint文件
    checkpoint_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # 加载checkpoint
    generator_model, discriminator_model, epoch = load_model_checkpoint(generator_model, discriminator_model, checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 验证模型
    avg_loss = validate_model(generator_model, dataloader, criterion_depth, criterion_mask, device)
    print(f"Validation Loss: {avg_loss:.8f}")