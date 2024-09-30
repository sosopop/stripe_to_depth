import os
import shutil

# 指定源目录和目标目录
source_directory = 'datasets'
target_directory = 'output_datasets'

# 确保目标目录存在
os.makedirs(target_directory, exist_ok=True)

# 用于生成新文件编号
file_counter = 1

# 遍历源目录中的所有子目录
for subdir, _, files in os.walk(source_directory):
    # 获取当前子目录中的所有文件，按文件名排序，确保顺序一致
    for file in sorted(files):
        # 过滤出需要处理的文件类型
        if file.endswith(('_depth.png', '_image.png', '_mask.png', '_z.tiff')):
            # 提取文件名的后缀部分
            suffix = file.split('_', 1)[1]
            # 生成新的文件名前缀
            new_file_name = f"{file_counter:08}_{suffix}"
            # 构建源文件和目标文件的路径
            source_file_path = os.path.join(subdir, file)
            target_file_path = os.path.join(target_directory, new_file_name)
            # 复制文件到目标目录并重命名
            shutil.copy2(source_file_path, target_file_path)
            print(f"Copied {source_file_path} to {target_file_path}")
            
            # 如果当前文件是以'_z.tiff'结尾的文件，表示同一组数据的结束，可以更新编号
            if file.endswith('_z.tiff'):
                file_counter += 1

print("所有文件已重新编号并拷贝到目标目录。")
