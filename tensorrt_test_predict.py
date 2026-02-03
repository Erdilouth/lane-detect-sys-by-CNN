import cv2
import numpy as np
import tensorrt as trt
import os
from cuda.bindings import driver, runtime as cuda_runtime

# 1. 环境初始化
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)


def lane_detection_pipeline(image_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)

    # --- 加载模型 ---
    with open(engine_path, "rb") as f:
        trt_runtime = trt.Runtime(logger)
        engine = trt_runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # --- 2. 图像预处理 (OpenCV) ---
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"无法读取图片: {image_path}");
        return

    h_orig, w_orig = raw_img.shape[:2]
    # 尺寸缩放并转为 RGB (模型训练通常用 RGB)
    input_img = cv2.resize(raw_img, (512, 256))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # 归一化: (0-255) -> (0-1) -> Mean/Std (这里按常见的 MobileNet 归一化)
    input_img = input_img.astype(np.float32) / 255.0
    # HWC -> CHW
    input_img = input_img.transpose(2, 0, 1)
    # 增加 Batch 维度
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.ascontiguousarray(input_img)

    # --- 3. 显存准备 ---
    input_name = "input"
    output_name = "output"
    input_shape = (1, 3, 256, 512)

    # 设置输入形状以解决之前负数的问题
    context.set_input_shape(input_name, input_shape)
    actual_output_shape = context.get_tensor_shape(output_name)

    in_nbytes = trt.volume(input_shape) * 4
    out_nbytes = trt.volume(actual_output_shape) * 4

    _, d_input = cuda_runtime.cudaMalloc(in_nbytes)
    _, d_output = cuda_runtime.cudaMalloc(out_nbytes)

    # --- 4. 执行推理 ---
    # H2D 拷贝
    cuda_runtime.cudaMemcpy(d_input, input_img.ctypes.data, in_nbytes,
                            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # 绑定地址 (针对 v8/v9 的 execute_v2)
    # 注意：如果 v10+ 且用 execute_v2，API 逻辑一致
    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)

    print("🚀 正在执行车道线检测...")
    # 兼容性修复：尝试 execute_v2
    try:
        context.execute_v2([int(d_input), int(d_output)])
    except AttributeError:
        # 如果是极新的 TensorRT 10 但没有 v3，尝试通用的同步执行
        context.execute_v2(bindings=[int(d_input), int(d_output)])

    # D2H 拷贝回结果
    host_output = np.empty(actual_output_shape, dtype=np.float32)
    cuda_runtime.cudaMemcpy(host_output.ctypes.data, d_output, out_nbytes,
                            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # --- 5. 后处理与可视化 ---
    # 模型输出通常是 Sigmoid 后的概率 (1, 1, 256, 512)
    mask = host_output[0, 0]
    # 二值化 (阈值 0.5)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255

    # 将 Mask 缩放回原图尺寸
    mask_resized = cv2.resize(binary_mask, (w_orig, h_orig))

    # 创建一个紫色的覆盖层
    color_mask = np.zeros_like(raw_img)
    color_mask[mask_resized > 0] = [255, 0, 255]  # BGR 紫色

    # 融合原图
    result = cv2.addWeighted(raw_img, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Lane Detection Result", result)
    cv2.imwrite("result_lane.jpg", result)
    print("✅ 检测完成！结果已保存至 result_lane.jpg")
    cv2.waitKey(0)

    # 释放资源
    cuda_runtime.cudaFree(d_input)
    cuda_runtime.cudaFree(d_output)


if __name__ == "__main__":
    lane_detection_pipeline("D:/Database for ML/archive/CULane/video_example/05081544_0305/05081544_0305-000002.jpg", "mobilenet_lanenet.engine")