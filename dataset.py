import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CULaneDataset(Dataset):
    def __init__(self, data_root, list_file, img_size=(512, 256)):
        """
        :param data_root: 数据集根目录 (CULane 文件夹路径)
        :param list_file: 训练列表文件路径 (如 train_list.txt)
        :param img_size: 缩放后的尺寸 (宽, 高)，减小显存占用
        """
        self.data_root = data_root
        self.img_size = img_size

        # 读取列表文件中的所有图片相对路径
        with open(list_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 拼接完整的图片路径
        img_rel_path = self.samples[idx]
        img_path = os.path.join(self.data_root, img_rel_path)

        # 2. 读取原图并转换颜色通道
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"找不到图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 3. 生成空白标签掩膜 (Mask)
        # 初始化为全黑 (0)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 4. 解析 .txt 坐标并画线
        # CULane 的标签文件通常是 图片名.lines.txt
        label_path = img_path.replace('.jpg', '.lines.txt')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) < 2: continue

                    # 将坐标点转为 (N, 1, 2) 的 numpy 数组格式，方便 OpenCV 绘图
                    points = np.array(coords).reshape(-1, 2).astype(np.int32)
                    # 在 mask 上画出白色的线 (值设为 255)，线条厚度设为 5-10
                    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=8)

        # 5. 统一缩放尺寸 (Resize)
        # 这是为了让所有图片大小一致，方便模型批量处理
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 6. 数据格式转换 (转换为 PyTorch 需要的张量格式)
        # 图像：HWC -> CHW，归一化到 [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # 掩膜：归一化到 0 (背景) 和 1 (车道线)
        mask_tensor = torch.from_numpy(mask).long()
        mask_tensor[mask_tensor > 0] = 1

        return image_tensor, mask_tensor


print("Dataset 类已整合完成，包含坐标解析与缩放功能。")

# 1. 定义你电脑上的实际路径
my_data_path = 'D:/database/archive/CULane/driver_161_90frame/driver_161_90frame'
my_list_file = 'D:/Portfolio/lane-detect-sys-by-CNN/train_list.txt'

# 2. 创建数据集对象（此时才把路径传进去）
dataset = CULaneDataset(data_root=my_data_path, list_file=my_list_file)

# 3. 试着取出一张图看看
img, mask = dataset[0]
print(f"成功读取！图片形状为: {img.shape}")