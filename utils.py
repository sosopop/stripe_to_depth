import os
import matplotlib.pyplot as plt
import torch

def visualize_sample(image, depth_pred, mask_pred, depth_gt, mask_gt, epoch, save_dir='visualizations'):
    """
    可视化并保存原图、推理的掩码图、推理的归一化深度图、推理的掩码后的归一化深度图，
    以及对应的标注（ground truth）图像。
    
    参数:
    - image: 原图
    - depth_pred: 推理出的深度图
    - mask_pred: 推理出的掩码图
    - depth_gt: 标注的深度图
    - mask_gt: 标注的掩码图
    - epoch: 当前的训练轮次，用于命名保存的图像
    - save_dir: 保存图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 去掉批次维度
    image = image.squeeze(0)
    depth_pred = depth_pred.squeeze(0)
    mask_pred = mask_pred.squeeze(0)
    depth_pred = depth_pred * torch.sigmoid(mask_pred)
    depth_gt = depth_gt.squeeze(0)
    mask_gt = mask_gt.squeeze(0)

    mask_pred = mask_pred > 0
    mask_gt = mask_gt > 0

    # 深度图归一化处理（预测）
    z_pred_normalized = depth_pred
    z_pred_normalized = torch.clamp(z_pred_normalized, min=-1, max=0)
    z_pred_normalized = z_pred_normalized + 1
    z_pred_normalized = z_pred_normalized * (mask_gt)
    
    # 深度图归一化处理（标注）
    z_gt_normalized = depth_gt
    z_gt_normalized = torch.clamp(depth_gt, min=-1, max=0)
    z_gt_normalized = z_gt_normalized + 1
    
    # 掩码处理后的深度图归一化（标注）
    masked_z_gt = torch.where(mask_gt == 1, depth_gt, torch.nan)
    z_mask_gt_data = masked_z_gt[~torch.isnan(masked_z_gt)]
    z_mask_gt_max = torch.max(z_mask_gt_data)
    z_mask_gt_min = torch.min(z_mask_gt_data)
    z_mask_gt_normalized = (masked_z_gt - z_mask_gt_min) / (z_mask_gt_max - z_mask_gt_min)
    z_mask_gt_normalized[torch.isnan(z_mask_gt_normalized)] = 0
    
    # 掩码处理后的深度图归一化（预测）
    masked_z_pred = torch.where(mask_pred == 1, depth_pred, torch.nan)
    # z_mask_pred_data = masked_z_pred[~torch.isnan(masked_z_pred)]
    # z_mask_pred_max = torch.max(z_mask_pred_data)
    # z_mask_pred_min = torch.min(z_mask_pred_data)
    masked_z_pred = masked_z_pred.clamp(min=z_mask_gt_min, max=z_mask_gt_max)
    z_mask_pred_normalized = (masked_z_pred - z_mask_gt_min) / (z_mask_gt_max - z_mask_gt_min)
    z_mask_pred_normalized[torch.isnan(z_mask_pred_normalized)] = 0

    # 使用matplotlib显示并保存图像
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 原图
    axes[0, 0].imshow(image.cpu().detach(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 推理的掩码图
    axes[0, 1].imshow(mask_pred.cpu().detach(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted Mask')
    axes[0, 1].axis('off')

    # 推理的归一化深度图
    axes[0, 2].imshow(z_pred_normalized.cpu().detach(), cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted Normalized Depth')
    axes[0, 2].axis('off')

    # 推理的掩码后的归一化深度图
    axes[0, 3].imshow(z_mask_pred_normalized.cpu().detach(), cmap='jet', vmin=0, vmax=1)
    axes[0, 3].set_title('Masked & Normalized Depth Prediction')
    axes[0, 3].axis('off')

    # 标注的掩码图
    axes[1, 1].imshow(mask_gt.cpu().detach(), cmap='gray')
    axes[1, 1].set_title('Ground Truth Mask')
    axes[1, 1].axis('off')

    # 标注的归一化深度图
    axes[1, 2].imshow(z_gt_normalized.cpu().detach(), cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title('Ground Truth Normalized Depth')
    axes[1, 2].axis('off')

    # 标注的掩码后的归一化深度图
    axes[1, 3].imshow(z_mask_gt_normalized.cpu().detach(), cmap='jet', vmin=0, vmax=1)
    axes[1, 3].set_title('Ground Truth Masked & Normalized Depth')
    axes[1, 3].axis('off')

    # 移除未使用的子图
    fig.delaxes(axes[1, 0])

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

def save_model_checkpoint(model, epoch, save_dir='checkpoints'):
    """
    保存模型检查点到指定目录
    
    参数:
    - model: 要保存的模型
    - epoch: 当前的训练轮次
    - save_dir: 保存检查点的目录，默认为 'checkpoints'
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建保存路径
    save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")
    
def load_model_checkpoint(model, load_path):
    """
    加载模型检查点

    参数:
    - model: 要加载的模型实例
    - load_path: 检查点的路径

    返回:
    - model: 加载了检查点的模型
    """

    # 加载模型状态字典
    model.load_state_dict(torch.load(load_path))

    return model


def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))