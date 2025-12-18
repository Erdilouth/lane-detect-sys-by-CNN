import os
import random

# 1. 设置你的数据集根目录
# 比如你的图片在 D:/CULane/driver_161_90frame...
dataset_root = 'D:/database/archive/CULane/driver_161_90frame/driver_161_90frame'


def generate_list():
    all_images = []

    # 遍历文件夹查找所有 .jpg 文件
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                # 获取相对路径，例如: driver_161_90frame/06030819_0755.MP4/00000.jpg
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, dataset_root)
                all_images.append(relative_path)

    # 打乱顺序（为了让模型学习更均匀）
    random.shuffle(all_images)

    # 按照 8:2 的比例拆分为训练集和验证集
    split_idx = int(len(all_images) * 0.8)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    # 保存到文件
    with open('train_list.txt', 'w') as f:
        for item in train_list:
            f.write(item + '\n')

    with open('val_list.txt', 'w') as f:
        for item in val_list:
            f.write(item + '\n')

    print(f"成功！生成的训练样本数: {len(train_list)}, 验证样本数: {len(val_list)}")


if __name__ == "__main__":
    generate_list()