import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import DepthEstimationDataset2 as DepthEstimationDataset
from model_unet import UNet, Discriminator
from utils import visualize_sample, save_model_checkpoint, load_model_checkpoint, log_cosh_loss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 使用标签数据进行监督训练
def supervised_train(image, depth_gt, mask_gt, generator_model, generator_optimizer, criterion_depth, criterion_mask):
    # 前向传播
    generator_optimizer.zero_grad()
    output = generator_model(image)
    depth_pred = output[:, 0:1, :, :]  # 获取第一个通道，深度图
    mask_pred = output[:, 1:2, :, :]   # 获取第二个通道，掩码图
    
    # depth_mask_gt = depth_gt * mask_gt  # 掩码处理后的标注深度图
    # depth_mask_pred = depth_pred * mask_gt  # 掩码处理后的深度图
    
    # 计算损失
    loss_depth = criterion_depth(depth_pred, depth_gt)
    # loss_depth2 = criterion_depth(depth_pred * mask_gt, depth_gt * mask_gt)
    # loss_depth = criterion_depth(depth_mask_pred, depth_mask_gt)
    loss_mask = criterion_mask(mask_pred, mask_gt)
    # loss = loss_depth * 0.4 + loss_depth2 * 10 + loss_mask * 0.1
    loss = loss_depth * 0.9 + loss_mask * 0.1

    # 反向传播和优化
    loss.backward()
    generator_optimizer.step()

    # 统计损失
    return loss.item(), depth_pred, mask_pred

# 使用GAN进行半监督微调训练
def unsupervised_train(
    labeled_image, labeled_depth_pred, labeled_mask_pred, unlabeled_image,
    discriminator_model, generator_model, generator_optimizer, criterion_discriminator, discriminator_optimizer):
    
    # 如果不使用GAN进行训练，则直接返回 0.0
    # return 0.0, 0.0

    # 训练判别器
    discriminator_optimizer.zero_grad()
    
    # 生成无标签的深度图和掩码图
    output = generator_model(unlabeled_image)
    unlabeled_depth_pred = output[:, 0:1, :, :]  # 获取第一个通道，深度图
    unlabeled_mask_pred = output[:, 1:2, :, :]   # 获取第二个通道，掩码图

    # 将有标签数据和预测结果拼接.
    noise_weight = 0.8  # 噪声权重
    labeled_image_noise = labeled_image + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    labeled_pred_noise = labeled_depth_pred + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    labeled_input = torch.cat((labeled_image_noise, labeled_mask_pred, labeled_pred_noise), dim=1)
    # 将无标签数据和预测结果拼接
    unlabeled_image_noise = unlabeled_image + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    unlabeled_pred_noise = unlabeled_depth_pred + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    unlabeled_input = torch.cat((unlabeled_image_noise, unlabeled_mask_pred, unlabeled_pred_noise), dim=1)
    
    # 判别器对有标签样本进行预测 (真实样本)
    labeled_output = discriminator_model(labeled_input.detach())
    # 判别器对无标签样本进行预测 (生成样本)
    unlabeled_output = discriminator_model(unlabeled_input.detach())
    
    # 创建真实标签 (1 表示真实)
    real_labels = torch.ones(labeled_output.size(), device=labeled_output.device)
    # 创建生成标签 (0 表示生成)
    fake_labels = torch.zeros(unlabeled_output.size(), device=unlabeled_output.device)
    
    # 计算判别器损失，针对有标签数据 (真实) 和无标签数据 (生成)
    loss_real = criterion_discriminator(labeled_output, real_labels)
    loss_fake = criterion_discriminator(unlabeled_output, fake_labels)
    loss_discriminator = (loss_real + loss_fake) * 0.5
    
    # 反向传播和更新判别器
    loss_discriminator.backward()
    discriminator_optimizer.step()
    
    # 训练生成器 (使生成器生成的样本尽可能被判别器认为是真实的)
    generator_optimizer.zero_grad()
    
    # 重新生成无标签的深度图和掩码图
    output = generator_model(unlabeled_image)
    unlabeled_depth_pred = output[:, 0:1, :, :]  # 获取第一个通道，深度图
    unlabeled_mask_pred = output[:, 1:2, :, :]   # 获取第二个通道，掩码图

    # 拼接无标签的预测结果，作为判别器输入
    unlabeled_image_noise = unlabeled_image + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    unlabeled_pred_noise = unlabeled_depth_pred + torch.randn_like(labeled_image) * noise_weight  # 加入噪声
    unlabeled_input = torch.cat((unlabeled_image_noise, unlabeled_mask_pred, unlabeled_pred_noise), dim=1)
    
    # 判别器对无标签的生成样本进行预测
    unlabeled_output = discriminator_model(unlabeled_input)
    
    # 生成器希望判别器认为生成的样本是真实的 (标签为 1)
    loss_generator = criterion_discriminator(unlabeled_output, real_labels)
    
    # 反向传播和更新生成器
    loss_generator.backward()
    generator_optimizer.step()
    
    # 返回生成器和判别器的损失
    return loss_generator.item(), loss_discriminator.item()

def train_model(
    generator_model, discriminator_model, 
    labeled_train_dataloader, unlabeled_train_dataloader, val_dataloader, 
    criterion_depth, criterion_mask, criterion_discriminator, 
    optimizer, discriminator_optimizer, generator_optimizer, num_epochs=25, start_epoch = 0, device='cuda'):
    
    writer = SummaryWriter(log_dir='runs/gan_training')
    
    generator_model = generator_model.to(device)
    discriminator_model = discriminator_model.to(device)
    
    best_val_loss = float('inf')  # 初始化最优验证损失

    for epoch in range(start_epoch, num_epochs):
        generator_model.train()
        discriminator_model.train()
        
        supervised_running_loss = 0.0
        discriminator_running_loss = 0.0
        generator_running_loss = 0.0
        
        unsupervised_iter = iter(unlabeled_train_dataloader)
        
        progress_bar = tqdm(enumerate(labeled_train_dataloader), total=len(labeled_train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (image, depth_gt, mask_gt) in progress_bar:
            unlabeled_image, _, _ = next(unsupervised_iter)
            if image.shape[0] != unlabeled_image.shape[0]:
                break
            
            # 将数据移动到 GPU
            unlabeled_image = unlabeled_image.to(device)
            image = image.to(device)
            depth_gt = depth_gt.to(device)
            mask_gt = mask_gt.to(device)
            
            # 使用标签数据进行监督训练
            loss, depth_pred, mask_pred = supervised_train(image, depth_gt, mask_gt, generator_model, optimizer, criterion_depth, criterion_mask)
            supervised_running_loss += loss
            
            # 使用GAN进行半监督微调训练
            if epoch > 30:  # 开始使用GAN进行训练
                generator_loss, discriminator_loss = unsupervised_train(image, depth_pred.detach(), mask_pred.detach(), unlabeled_image, discriminator_model, generator_model, generator_optimizer, criterion_discriminator, discriminator_optimizer)
                discriminator_running_loss += discriminator_loss
                generator_running_loss += generator_loss
            else:
                generator_loss, discriminator_loss = 0.0, 0.0
            
            # 更新进度条描述
            progress_bar.set_postfix(
                {
                    "S Loss": f"{supervised_running_loss / (batch_idx + 1):.10f}", 
                    "D Loss": f"{discriminator_running_loss / (batch_idx + 1):.10f}",
                    "G Loss": f"{generator_running_loss / (batch_idx + 1):.10f}"
                })
            
        writer.add_scalar('Loss/S', supervised_running_loss / len(labeled_train_dataloader), epoch)
        writer.add_scalar('Loss/D', discriminator_running_loss / len(labeled_train_dataloader), epoch)
        writer.add_scalar('Loss/G', generator_running_loss / len(labeled_train_dataloader), epoch)

        # 每 50 轮进行一次验证和保存模型
        if (epoch + 1) % 1 == 0:
            generator_model.eval()  # 切换到评估模式
            val_running_loss = 0.0
            with torch.no_grad():
                for image, depth_gt, mask_gt in val_dataloader:
                    image = image.to(device)
                    depth_gt = depth_gt.to(device)
                    mask_gt = mask_gt.to(device)
                    
                    output = generator_model(image)
                    depth_pred = output[:, 0:1, :, :]
                    mask_pred = output[:, 1:2, :, :]
                    
                    # depth_mask_gt = depth_gt * mask_gt  # 掩码处理后的标注深度图
                    # depth_mask_pred = depth_pred * mask_gt  # 掩码处理后的深度图
                    
                    # loss_depth = criterion_depth(depth_pred, depth_gt)
                    loss_depth = criterion_depth(depth_pred, depth_gt)
                    loss_mask = criterion_mask(mask_pred, mask_gt)
                    loss = loss_depth * 0.9 + loss_mask * 0.1

                    val_running_loss += loss.item()
                    
            val_loss = val_running_loss / len(val_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.10f}')

            writer.add_scalar('Loss/V', val_loss, epoch)
            
            # 保存当前最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_checkpoint(generator_model, discriminator_model, epoch)  # 保存模型
                print(f"Best model saved with Validation Loss: {best_val_loss:.10f}")

            # 可视化结果
            with torch.no_grad():
                for image, depth, mask in val_dataloader:
                    image = image.to(device)
                    output = generator_model(image)
                    depth_pred = output[0, 0:1, :, :]
                    mask_pred = output[0, 1:2, :, :]
                    depth = depth[0]
                    mask = mask[0]

                    # 可视化并保存结果
                    visualize_sample(image[0].cpu(), depth_pred.cpu(), mask_pred.cpu(), depth.cpu(), mask.cpu(), epoch + 1)
                    break  # 只展示一个批次

        generator_model.train()  # 切换回训练模式
        
    return generator_model


if __name__ == '__main__':
    # 加载数据集
    datasets_dir = 'datasets3'
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transforms.RandomErasing(p=0.8, scale=(0.07, 0.07), ratio=(0.5, 2), value=0),
        transforms.RandomErasing(p=0.8, scale=(0.05, 0.05), ratio=(0.5, 2), value=0),
        transforms.RandomErasing(p=0.8, scale=(0.1, 0.1), ratio=(0.5, 2), value=0)
    ])
    labeled_train_dataset = DepthEstimationDataset(root_dir=f'{datasets_dir}/supervised_train', data_size=5000, transform=transform, noise_mask_prob=(0.0, 0.03))
    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=8, shuffle=True, num_workers=4)
    unlabeled_train_dataset = DepthEstimationDataset(root_dir=f'{datasets_dir}/unsupervised_train', data_size=5000, transform=transform, noise_mask_prob=(0.0, 0.03))
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    val_dataset = DepthEstimationDataset(root_dir=f'{datasets_dir}/val')
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"Supervised training set size: {len(labeled_train_dataset)}, Unsupervised training set size: {len(unlabeled_train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 初始化模型
    generator_model = UNet(input_channels=1, output_channels=2, complexity=8, use_transformer=False)
    discriminator_model = Discriminator(complexity=4, input_channels=3)
    
    # 获取最新的checkpoint文件
    checkpoint_dir = 'checkpoints'
    epoch = 0
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint file: {checkpoint_path}")
            # 加载checkpoint
            generator_model, discriminator_model, epoch = load_model_checkpoint(generator_model, discriminator_model, checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 定义损失函数和优化器
    criterion_depth = nn.MSELoss()  # 深度图损失
    # criterion_depth = log_cosh_loss  # 使用 log_cosh_loss 代替 MSELoss
    criterion_mask = nn.BCEWithLogitsLoss()  # 掩码图损失
    criterion_discriminator = torch.nn.BCELoss()  # 判别器损失
    
    # 监督学习优化器
    optimizer = optim.Adam(generator_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # GAN优化器
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=1e-4)
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=1e-5, betas=(0.5, 0.999))
    
    # 训练模型
    trained_model = train_model(
        generator_model, discriminator_model, 
        labeled_train_dataloader, unlabeled_train_dataloader, val_dataloader, 
        criterion_depth, criterion_mask, criterion_discriminator, 
        optimizer, discriminator_optimizer, generator_optimizer, num_epochs=100000, start_epoch=epoch, device='cuda')

    # 保存模型
    torch.save(trained_model.state_dict(), 'depth_estimation_model.pth')
