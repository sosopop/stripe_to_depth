import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import DepthEstimationDataset
from model_unet import UNet
from utils import visualize_sample, save_model_checkpoint

# 定义训练函数
def train_model(model, dataloader, criterion_depth, criterion_mask, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (image, depth_gt, mask_gt) in enumerate(dataloader):
            # 将数据移动到 GPU
            image = image.to(device)
            depth_gt = depth_gt.to(device)
            mask_gt = mask_gt.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(image)
            depth_pred = output[:, 0:1, :, :]  # 获取第一个通道，深度图
            mask_pred = output[:, 1:2, :, :]   # 获取第二个通道，掩码图
            
            # depth_mask_pred = torch.where(mask_pred > 0, depth_pred, torch.zeros_like(depth_pred))  # 掩码处理后的深度图
            # depth_mask_gt = torch.where(mask_gt > 0, depth_gt, torch.zeros_like(depth_gt))  # 掩码处理后的标注深度图
            
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

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 每 20 轮可视化并保存图像
        if (epoch + 1) % 100 == 0:
            model.eval()  # 切换到评估模式
            with torch.no_grad():
                for image, depth, mask in dataloader:
                    image = image.to(device)
                    output = model(image)
                    depth_pred = output[0, 0:1, :, :]
                    mask_pred = output[0, 1:2, :, :]
                    depth = depth[0]
                    mask = mask[0]

                    # 可视化并保存结果
                    visualize_sample(image[0].cpu(), depth_pred.cpu(), mask_pred.cpu(), depth.cpu(), mask.cpu(), epoch + 1)
                    break  # 只展示一个批次

            # 保存模型检查点
            save_model_checkpoint(model, epoch)

            model.train()  # 切换回训练模式
            
    return model


if __name__ == '__main__':
    # 加载数据集
    dataset = DepthEstimationDataset(root_dir='data')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    # 初始化模型
    model = UNet(input_channels=1, output_channels=2, complexity=8)

    # 定义损失函数和优化器
    criterion_depth = nn.MSELoss()  # 深度图损失
    criterion_mask = nn.BCEWithLogitsLoss()  # 掩码图损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练模型
    trained_model = train_model(model, dataloader, criterion_depth, criterion_mask, optimizer, num_epochs=10000)

    # 保存模型
    torch.save(trained_model.state_dict(), 'depth_estimation_model.pth')
