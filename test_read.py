import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 1. 读取深度文件 z.csv
z_data = pd.read_csv(r'data\4_0\z.csv', header=None)
z_array = z_data.values

# 2. 读取 mask.csv (二值位图)
mask_data = pd.read_csv(r'data\4_0\mask.csv', header=None)
mask_array = mask_data.values

# 3. 对 z.csv 进行掩码处理（将 mask 为 0 的区域设为 NaN）
masked_z = np.where(mask_array == 1, z_array, np.nan)

# 4. 归一化掩码后的深度图数据到 [0, 255]，仅归一化有效的（mask为1）的部分
valid_values = masked_z[~np.isnan(masked_z)]
z_min, z_max = valid_values.min(), valid_values.max()
normalized_z = (masked_z - z_min) / (z_max - z_min) * 255
normalized_z[np.isnan(normalized_z)] = 0  # 将无效区域（mask为0）的值设为0

# 无效部分
invalid_mask = np.where(mask_array == 1, np.nan, z_array)
invalid_z = invalid_mask[~np.isnan(invalid_mask)]
z_min, z_max = invalid_z.min(), invalid_z.max()

# 5. 转换为8位灰度图
grayscale_image = normalized_z.astype(np.uint8)

# 6. 将 0 和 1 转换为二值掩码图像（0 -> 黑色，1 -> 白色）
mask_image = (mask_array * 255).astype(np.uint8)

# 7. 显示掩码后的灰度图像和二值掩码
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 显示掩码后的灰度图像
ax[0].imshow(grayscale_image, cmap='gray')
ax[0].set_title('Masked Grayscale Image')
ax[0].axis('off')

# 显示二值掩码图像
ax[1].imshow(mask_image, cmap='gray')
ax[1].set_title('Binary Mask Image')
ax[1].axis('off')

plt.show()

# 8. 保存掩码后的深度图像
cv2.imwrite('masked_grayscale_image.png', grayscale_image)
