import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import random
import tifffile as tiff

def process_dataset(source_root, target_dir, supervised_ratio, unsupervised_ratio, test_ratio, val_ratio):
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
    
    # 收集所有满足条件的目录
    directories = []
    
    for root, dirs, files in os.walk(source_root):
        if 'f.png' in files and 'z.png' in files and 'm.png' in files:
            directories.append(root)
    
    # 数据洗牌
    random.shuffle(directories)
    
    # 总数据量
    total_dirs = len(directories)

    # 确保比例和为1
    assert abs(supervised_ratio + unsupervised_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "划分比例之和必须为1"

    # 根据外部传入的比例分配数据
    supervised_count = int(total_dirs * supervised_ratio)
    unsupervised_count = int(total_dirs * unsupervised_ratio)
    test_count = int(total_dirs * test_ratio)
    val_count = total_dirs - supervised_count - unsupervised_count - test_count  # 剩余作为 val
    
    supervised_dirs = directories[:supervised_count]
    unsupervised_dirs = directories[supervised_count:supervised_count + unsupervised_count]
    test_dirs = directories[supervised_count + unsupervised_count:supervised_count + unsupervised_count + test_count]
    val_dirs = directories[supervised_count + unsupervised_count + test_count:]
    
    # 处理函数：将图像直接拷贝到目标文件夹，并处理z.png归一化
    def process_and_save_data(root, target_sub_dir):
        nonlocal file_counter
        # 使用8位数字作为文件名前缀
        base_name = f'{file_counter:08d}'
        
        # 构建源文件路径
        image_path = os.path.join(root, 'f.png')
        z_path = os.path.join(root, 'z.png')
        mask_path = os.path.join(root, 'm.png')
        
        try:
            # 生成目标文件路径，使用新的命名方式
            target_image_path = os.path.join(target_sub_dir, f'{base_name}_image.png')
            target_mask_path = os.path.join(target_sub_dir, f'{base_name}_mask.png')
            target_z_path = os.path.join(target_sub_dir, f'{base_name}_z.tiff')
            target_depth_image_path = os.path.join(target_sub_dir, f'{base_name}_depth.png')
            
            # 拷贝图像文件
            shutil.copyfile(image_path, target_image_path)
            shutil.copyfile(mask_path, target_mask_path)
            shutil.copyfile(z_path, target_depth_image_path)
            
            # 加载并处理深度图像z.png
            z_image = cv2.imread(z_path, cv2.IMREAD_UNCHANGED)
            if z_image is not None:
                # 使用简单的除以255归一化
                z_image_normalized = z_image / 255.0
                z_image_normalized = z_image_normalized.astype(np.float32)
                # 将归一化的z图像保存为tiff文件
                tiff.imwrite(target_z_path, z_image_normalized)
            else:
                print(f"跳过无效的深度图像 {root}")
                
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
    source_root = r"D:\code\stripe_to_depth\backup\test"  # 替换为您的源数据集路径
    target_dir = r"datasets3"  # 替换为您的目标数据集路径
    
    # 从外部传入的划分比例
    supervised_ratio = 0.4  # 40%
    unsupervised_ratio = 0.4  # 40%
    test_ratio = 0.1  # 10%
    val_ratio = 0.1  # 10%
    
    process_dataset(source_root, target_dir, supervised_ratio, unsupervised_ratio, test_ratio, val_ratio)
