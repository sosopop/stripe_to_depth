import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets2 import DepthEstimationDataset2 as DepthEstimationDataset
from model_unet import UNet
from utils import visualize_sample, save_model_checkpoint, load_model_checkpoint, log_cosh_loss

def train_model(model, train_dataloader, val_dataloader, criterion_depth, criterion_mask, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')  # 初始化最优验证损失

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (image, depth_gt, mask_gt) in progress_bar:
            # 将数据移动到 GPU
            image = image.to(device)
            depth_gt = depth_gt.to(device)
            mask_gt = mask_gt.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(image)
            depth_pred = output[:, 0:1, :, :]  # 获取第一个通道，深度图
            mask_pred = output[:, 1:2, :, :]   # 获取第二个通道，掩码图
            
            depth_mask_gt = depth_gt * mask_gt  # 掩码处理后的标注深度图
            depth_mask_pred = depth_pred * torch.sigmoid(mask_pred)  # 掩码处理后的深度图
            
            # 计算损失
            loss_depth = criterion_depth(depth_mask_gt, depth_mask_pred)
            loss_mask = criterion_mask(mask_pred, mask_gt)
            loss = loss_depth * 0.9 + loss_mask * 0.1

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            
            # 更新进度条描述
            progress_bar.set_postfix({"Loss": f"{running_loss / (batch_idx + 1):.10f}"})

        # 每 5 轮进行一次验证和保存模型
        if (epoch + 1) % 5 == 0:
            model.eval()  # 切换到评估模式
            val_running_loss = 0.0
            with torch.no_grad():
                for image, depth_gt, mask_gt in val_dataloader:
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

                    val_running_loss += loss.item()
                    
            val_loss = val_running_loss / len(val_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.10f}')

            # 保存当前最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_checkpoint(model, epoch)  # 保存模型
                print(f"Best model saved with Validation Loss: {best_val_loss:.10f}")

            # 可视化结果
            with torch.no_grad():
                for image, depth, mask in val_dataloader:
                    image = image.to(device)
                    output = model(image)
                    depth_pred = output[0, 0:1, :, :]
                    mask_pred = output[0, 1:2, :, :]
                    depth = depth[0]
                    mask = mask[0]

                    # 可视化并保存结果
                    visualize_sample(image[0].cpu(), depth_pred.cpu(), mask_pred.cpu(), depth.cpu(), mask.cpu(), epoch + 1)
                    break  # 只展示一个批次

        model.train()  # 切换回训练模式
        
    return model


if __name__ == '__main__':
    # 加载数据集
    train_dataset = DepthEstimationDataset(root_dir='datasets/train')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_dataset = DepthEstimationDataset(root_dir='datasets/train')
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=1)
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # 初始化模型
    model = UNet(input_channels=1, output_channels=2, complexity=8)
    
    # 获取最新的checkpoint文件
    checkpoint_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint file: {checkpoint_path}")
            # 加载checkpoint
            model = load_model_checkpoint(model, checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 定义损失函数和优化器
    criterion_depth = nn.MSELoss()  # 深度图损失
    # criterion_depth = log_cosh_loss  # 深度图损失
    # criterion_depth = piq.SSIMLoss()
    criterion_mask = nn.BCEWithLogitsLoss()  # 掩码图损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练模型
    trained_model = train_model(model, train_dataloader, val_dataloader,criterion_depth, criterion_mask, optimizer, num_epochs=100000)

    # 保存模型
    torch.save(trained_model.state_dict(), 'depth_estimation_model.pth')
