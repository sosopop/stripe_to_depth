import os
import shutil
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
import random

def process_dataset(source_root, target_dir):
    # 确保目标目录和子目录存在
    supervised_dir = os.path.join(target_dir, 'supervised_train')
    unsupervised_dir = os.path.join(target_dir, 'unsupervised_train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(supervised_dir, exist_ok=True)
    os.makedirs(unsupervised_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 用于生成连续的文件编号
    file_counter = 1
    
    # 目标图像的尺寸
    target_size = (576, 576)
    
    # 收集所有满足条件的目录
    directories = []
    
    for root, dirs, files in os.walk(source_root):
        if '1.png' in files and 'z.csv' in files and 'mask.csv' in files:
            directories.append(root)
    
    # 数据洗牌
    random.shuffle(directories)
    
    # 根据比例分配数据
    total_dirs = len(directories)
    supervised_count = (total_dirs * 2) // 6  # 2/5
    unsupervised_count = (total_dirs * 2) // 6  # 2/5
    test_count = total_dirs // 6  # 1/5
    val_count = total_dirs - supervised_count - unsupervised_count - test_count  # 剩余作为 val
    
    supervised_dirs = directories[:supervised_count]
    unsupervised_dirs = directories[supervised_count:supervised_count + unsupervised_count]
    test_dirs = directories[supervised_count + unsupervised_count:supervised_count + unsupervised_count + test_count]
    val_dirs = directories[supervised_count + unsupervised_count + test_count:]
    
    # 处理函数
    def process_and_save_data(root, target_sub_dir):
        nonlocal file_counter
        # if root.find('42_') != -1:  # 过滤掉 42_ 开头的文件夹
        #     return
        # 构建源文件路径
        image_path = os.path.join(root, '1.png')
        z_path = os.path.join(root, 'z.csv')
        mask_path = os.path.join(root, 'mask.csv')
        
        # 生成新的文件名（使用8位数字，从00000001开始）
        base_name = f'{file_counter:08d}'
        
        # 构建目标文件路径
        target_image_path = os.path.join(target_sub_dir, f'{base_name}_image.png')
        target_mask_path = os.path.join(target_sub_dir, f'{base_name}_mask.png')
        target_z_path = os.path.join(target_sub_dir, f'{base_name}_z.tiff')
        target_depth_image_path = os.path.join(target_sub_dir, f'{base_name}_depth.png')
        
        try:
            # 复制并处理图像文件
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # 获取原始图像的尺寸
            original_height, original_width = image.shape[:2]
            
            # 计算需要扩展的边缘大小
            top = (target_size[1] - original_height) // 2
            bottom = target_size[1] - original_height - top
            left = (target_size[0] - original_width) // 2
            right = target_size[0] - original_width - left
            
            # 添加黑边扩展图像
            image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            
            # 保存扩展后的图像
            cv2.imwrite(target_image_path, image_with_border)
            
            # 处理深度图数据
            z_data = pd.read_csv(z_path, header=None).values
            z_data = z_data.astype(np.float32)  # 确保数据类型为32位浮点数
            z_data_with_border = cv2.copyMakeBorder(z_data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[-1])
            tiff.imwrite(target_z_path, z_data_with_border)
            
            # 归一化 z_data 并保存深度图像
            z_data_with_border[z_data_with_border < -0.4] = -1  # 过滤掉异常值
            z_data_with_border[z_data_with_border > -0.3] = -1  # 过滤掉异常值
            z_data_with_border[image_with_border<2] = -1
            z_data_with_border = np.clip(z_data_with_border, -0.4, -0.3)
            min_z = z_data_with_border[z_data_with_border != -0.4].min()
            max_z = z_data_with_border[z_data_with_border != -0.4].max()
            z_data_with_border = ((np.clip(z_data_with_border, min_z, max_z) - min_z) / (max_z - min_z) * 255).astype(np.uint8)
            
            # 保存深度图像
            cv2.imwrite(target_depth_image_path, z_data_with_border)
            
            mask_data = z_data_with_border
            mask_data[z_data_with_border > 0] = 255  # 掩码中标记为 255 的像素对应于深度图像中大于 0 的像素
            # 保存掩码
            cv2.imwrite(target_mask_path, mask_data)
            
            file_counter += 1
            
        except Exception as e:
            print(f"处理目录 {root} 时发生错误: {str(e)}")
    
    # 使用 tqdm 添加进度条并处理数据
    for dir_type, sub_dirs, sub_dir_name in zip(
        ["Supervised", "Unsupervised", "Validation", "Test"],
        [supervised_dirs, unsupervised_dirs, val_dirs, test_dirs],
        [supervised_dir, unsupervised_dir, val_dir, test_dir]
    ):
        print(f"开始处理 {dir_type} 数据...")
        for root in tqdm(sub_dirs, desc=f"Processing {dir_type}"):
            process_and_save_data(root, sub_dir_name)
    
    print(f"数据集重组完成，共处理了 {file_counter-1} 组数据")

# 使用示例
if __name__ == "__main__":
    source_root = r"C:\Users\mengchao\Downloads\dataset"  # 替换为您的源数据集路径
    target_dir = r"D:\code\stripe_to_depth\datasets4"  # 替换为您的目标数据集路径
    
    process_dataset(source_root, target_dir)