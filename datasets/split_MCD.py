import os
import cv2
import pandas as pd
import numpy as np

def split_image(folder_name, training_mode):
    """将图片分割为多个小图像并保存到新文件夹中"""
    patch_H, patch_W = 512, 512  # image patch width and height
    overlap = 0  # overlap area
    input_path = f'./MCD/{folder_name}/'
    
    img_A_path = os.path.join(input_path, 'im1.jpg')
    img_B_path = os.path.join(input_path, 'im2.jpg')
    img_label_path = os.path.join(input_path, 'ref.png')
    
    img_A = cv2.imread(img_A_path, -1)
    img_B = cv2.imread(img_B_path, -1)
    img_label = cv2.imread(img_label_path, -1)

    # 检查图像是否成功加载
    if img_A is None or img_B is None or img_label is None:
        print(f"无法读取图像: {img_A_path}, {img_B_path}, {img_label_path}")
        return

    img_H, img_W, _ = img_A.shape
    
    print(f"处理文件夹: {input_path}")

    # 创建输出文件夹
    output_A_path = os.path.join(training_mode, 'A')
    output_B_path = os.path.join(training_mode, 'B')
    output_label_path = os.path.join(training_mode, 'label')
    
    os.makedirs(output_A_path, exist_ok=True)
    os.makedirs(output_B_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    if img_H > 512 and img_W > 512:
        for x in range(0, img_W, patch_W - overlap):
            for y in range(0, img_H, patch_H - overlap):
                x_str = x
                x_end = x + patch_W
                y_str = y
                y_end = y + patch_H
                
                # 检查子图像是否为 512x512
                if (x_end <= img_W) and (y_end <= img_H):
                    img_A_s = img_A[y_str:y_end, x_str:x_end, :]
                    img_B_s = img_B[y_str:y_end, x_str:x_end, :]
                    img_label_s = img_label[y_str:y_end, x_str:x_end]

                    # 生成文件名
                    image_name = f"{folder_name}_{y_str}_{y_end}_{x_str}_{x_end}.png"
                    save_img_A_s_path = os.path.join(output_A_path, image_name)
                    save_img_B_s_path = os.path.join(output_B_path, image_name)
                    save_img_label_s_path = os.path.join(output_label_path, image_name)

                    # 保存图像
                    if not os.path.isfile(save_img_A_s_path):
                        cv2.imwrite(save_img_A_s_path, img_A_s)
                    if not os.path.isfile(save_img_B_s_path):
                        cv2.imwrite(save_img_B_s_path, img_B_s)
                    if not os.path.isfile(save_img_label_s_path):
                        cv2.imwrite(save_img_label_s_path, img_label_s)
                    

def find_subfolder_names(parent_folder):
    """列出给定文件夹下的所有子文件夹名称"""
    subfolder_names = []
    
    # 遍历文件夹
    for root, dirs, _ in os.walk(parent_folder):
        for dir_name in dirs:
            subfolder_names.append(dir_name)  # 仅保存文件夹名称
    
    return subfolder_names

# 示例调用
mcd_folder = './MCD'  # 替换为你的 MCD 文件夹路径
subfolder_names = find_subfolder_names(mcd_folder)

# 输出结果
all_folders = []
for name in subfolder_names:
    print(name)
    all_folders.append(name)

# 随机分割数据
np.random.seed(42)  # 设置随机种子以保证可重复性
np.random.shuffle(all_folders)  # 随机打乱顺序

# 计算分割索引
total_count = len(all_folders)
train_size = int(total_count * 0.6)
val_size = int(total_count * 0.3)

# 划分数据
train_folders = all_folders[:train_size]
val_folders = all_folders[train_size:train_size + val_size]
test_folders = all_folders[train_size + val_size:]

# 输出结果
print("\n训练集 (60%):")
print(train_folders)
print("\n验证集 (30%):")
print(val_folders)
print("\n测试集 (10%):")
print(test_folders)

for name in train_folders:
    split_image(name, training_mode='train')
for name in val_folders:
    split_image(name, training_mode='val')
for name in test_folders:
    split_image(name, training_mode='test')


