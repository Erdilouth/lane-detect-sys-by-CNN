import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import MobileNetLaneNet  # 导入你的模型结构


def predict():
    # 1. 基本配置
    # 请确保这个路径是你刚才训练生成的权重文件
    model_path = 'lane_model_epoch_10.pth'
    test_img_path = 'D:/database/archive/CULane/driver_193_90frame/driver_193_90frame/06042016_0513.MP4/00000.jpg'
    img_size = (512, 256)  # 必须和训练时的尺寸一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型与权重
    model = MobileNetLaneNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式（关闭 BatchNorm 的更新）

    # 3. 读取并预处理图片
    original_img = cv2.imread(test_img_path)
    original_h, original_w = original_img.shape[:2]

    # 调整大小并归一化
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 增加 Batch 维度 [1, 3, 256, 512]

    # 4. 推理（让模型画线）
    with torch.no_grad():  # 推理时不需要计算梯度，节省内存
        output = model(img_tensor)

    # 将输出转回 numpy 格式
    mask = output.squeeze().cpu().numpy()  # [256, 512] 的概率图

    # 5. 后处理：二值化（概率大于 0.5 的认为是车道线）
    mask_binary = (mask > 0.5).astype(np.uint8) * 255

    # 将 mask 放大回原图尺寸，以便叠加
    mask_resized = cv2.resize(mask_binary, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 6. 将车道线叠加到原图上 (显示为红色)
    result = original_img.copy()
    result[mask_resized > 0] = [0, 0, 255]  # BGR 格式中的红色

    # 7. 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Detection Result")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    predict()