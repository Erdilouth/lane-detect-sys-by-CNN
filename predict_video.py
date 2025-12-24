import torch
import cv2
import numpy as np
from model import MobileNetLaneNet


def process_video(video_path, model_path):
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (512, 256)  # 必须与训练时保持一致

    # 2. 加载模型
    print(f"正在加载模型: {model_path}...")
    model = MobileNetLaneNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 开启评估模式

    # 3. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查路径。")
        return

    print("开始处理视频... 按 'q' 键退出。")

    # 4. 循环处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()  # ret是布尔值(是否读取成功)，frame是当前帧图片

        if not ret:
            print("视频播放结束。")
            break

        # --- 预处理 ---
        # 记录原图尺寸，方便最后还原
        original_h, original_w = frame.shape[:2]

        # OpenCV读入是BGR，模型训练通常用RGB，这里转换一下
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 缩放
        img_resized = cv2.resize(img_rgb, img_size)
        # 转Tensor并归一化
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

        # --- 模型推理 ---
        with torch.no_grad():
            output = model(img_tensor)

        # --- 后处理 ---
        # 提取结果并转回CPU
        mask = output.squeeze().cpu().numpy()
        # 二值化 (阈值 0.5)
        mask_binary = (mask > 0.5).astype(np.uint8)

        # 将预测的 Mask 放大回原视频帧的尺寸
        mask_resized = cv2.resize(mask_binary, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # --- 叠加显示 (修改版) ---
        # 1. 确保 mask 是布尔类型 (True/False)，方便索引
        # 注意：这里判断 > 0，因为之前我们乘了 255
        lane_mask = mask_resized > 0

        # 2. 创建一个全红色的图层
        color_mask = np.zeros_like(frame)
        color_mask[:, :] = [0, 0, 255]  # BGR格式：红色

        # 3. 先计算好 "原图 + 红色" 的混合效果 (整张图都混合)
        # alpha=1.0 (原图权重), beta=0.6 (红色权重), gamma=0
        blended_frame = cv2.addWeighted(frame, 1.0, color_mask, 0.6, 0)

        # 4. 只将 "车道线区域" 的像素替换为混合后的像素
        # 这样背景保持不变，只有车道线变色
        frame[lane_mask] = blended_frame[lane_mask]

        # --- 实时显示 ---
        cv2.imshow('Lane Detection Result', frame)

        # 按 'q' 键退出循环 (waitkey控制播放速度，1ms表示尽可能快)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 请修改为你的视频路径 (CULane里有很多 .MP4 文件)
    video_file = 'D:/database/archive/CULane/video_example/video_example.mp4'
    model_file = 'lane_model_epoch_10.pth'

    process_video(video_file, model_file)