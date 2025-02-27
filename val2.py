import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import DepthEstimationDataset2 as DepthEstimationDataset
from model_unet import UNet, Discriminator
from utils import visualize_sample, load_model_checkpoint, log_cosh_loss
import os

def validate_model(model, dataloader, criterion_depth, criterion_mask, device='cuda'):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_loss_depth = 0.0
    total_loss_mask = 0.0
    total_loss_depth2 = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (image, depth_gt, mask_gt) in enumerate(dataloader):
            image = image.to(device)
            depth_gt = depth_gt.to(device)
            mask_gt = mask_gt.to(device)

            output = model(image)
            depth_pred = output[:, 0:1, :, :]
            mask_pred = output[:, 1:2, :, :]

            loss_depth = criterion_depth(depth_pred, depth_gt)
            loss_mask = criterion_mask(mask_pred, mask_gt)
            loss_depth2 = criterion_depth(depth_pred[mask_gt > 0.5], depth_gt[mask_gt > 0.5])

            loss = loss_depth * 0.9 + loss_mask * 0.1
            total_loss += loss.item() * image.size(0)
            total_loss_depth += loss_depth.item() * image.size(0)
            total_loss_mask += loss_mask.item() * image.size(0)
            total_loss_depth2 += loss_depth2.item() * image.size(0)
            
            num_samples += image.size(0)

            for i in range(image.size(0)):
                visualize_sample(
                    image[i].cpu(),
                    depth_pred[i].cpu(),
                    mask_pred[i].cpu(),
                    depth_gt[i].cpu(),
                    mask_gt[i].cpu(),
                    f"validation_batch{batch_idx}_sample{i}",
                    save_dir="validation_samples"
                )

    avg_loss = total_loss / num_samples
    avg_loss_depth = total_loss_depth / num_samples
    avg_loss_mask = total_loss_mask / num_samples
    avg_loss_depth2 = total_loss_depth2 / num_samples

    return avg_loss, avg_loss_depth, avg_loss_mask, avg_loss_depth2

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = DepthEstimationDataset(root_dir='datasets3/test')
    # dataset = DepthEstimationDataset(root_dir='datasets3/val')
    # dataset = DepthEstimationDataset(root_dir='datasets3/unsupervised_train')
    # dataset = DepthEstimationDataset(root_dir='datasets3/supervised_train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化模型
    generator_model = UNet(input_channels=1, output_channels=2, complexity=8, use_transformer=False)
    discriminator_model = Discriminator(complexity=4, input_channels=3)

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
    avg_loss, avg_loss_depth, avg_loss_mask, avg_loss_depth2 = validate_model(generator_model, dataloader, criterion_depth, criterion_mask, device)
    print(f"Validation Loss: {avg_loss:.8f}, Depth Loss: {avg_loss_depth:.8f}, Mask Loss: {avg_loss_mask:.8f}, Depth2 Loss: {avg_loss_depth2:.8f}")