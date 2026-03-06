import cv2
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
import time
import os

# --- 全局配置 ---
ENGINE_PATH = "mobilenet_lanenet.engine"
INPUT_SHAPE = (1, 3, 256, 512)  # (Batch, Channel, Height, Width)
CAMERA_ID = 0  # UVC 摄像头通常为 0

# --- IPM (逆透视变换) 配置 ---
# TODO: 标定完成后，这里可以替换为真实的相机内外参计算出的变换矩阵
# 这里提供一个经验性的感兴趣区域 (ROI) 映射
SRC_POINTS = np.float32([
    [450, 460], [830, 460],  # 远端 左、右
    [150, 720], [1130, 720]  # 近端 左、右 (假设分辨率 1280x720)
])
DST_POINTS = np.float32([
    [300, 0], [980, 0],  # 鸟瞰图 远端
    [300, 720], [980, 720]  # 鸟瞰图 近端
])
IPM_MATRIX = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
IPM_INV_MATRIX = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)


class LaneDetectorTRT:
    """复用 video_inference.py 中的 TensorRT 引擎封装"""

    def __init__(self, engine_path):
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA 初始化失败: {err}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"找不到引擎文件: {engine_path}")

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input", INPUT_SHAPE)

        self.in_size = trt.volume(INPUT_SHAPE) * 4
        output_shape = self.context.get_tensor_shape("output")
        self.out_size = trt.volume(output_shape) * 4

        err, self.d_input = cudart.cudaMalloc(self.in_size)
        err, self.d_output = cudart.cudaMalloc(self.out_size)
        self.h_output = np.empty(output_shape, dtype=np.float32)

    def infer(self, img):
        # 预处理
        resized = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.ascontiguousarray(np.expand_dims(input_data, axis=0))

        # Host -> Device
        cudart.cudaMemcpy(self.d_input, input_data.ctypes.data, self.in_size,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 推理
        self.context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])

        # Device -> Host
        cudart.cudaMemcpy(self.h_output.ctypes.data, self.d_output, self.out_size,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # 二值化掩码提取
        mask = self.h_output[0, 0]
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        return cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    def cleanup(self):
        cudart.cudaFree(self.d_input)
        cudart.cudaFree(self.d_output)


def fit_lane_center(bev_mask):
    """从鸟瞰图掩码中提取车道线中心，并进行三次多项式拟合"""
    # 找到所有车道线像素点的坐标 (y, x)
    y_coords, x_coords = np.where(bev_mask > 0)

    if len(y_coords) < 50:  # 像素点太少，可能未检测到车道线
        return None, None

    # 计算每一行 (y) 的中心点 (x_mean)
    unique_y = np.unique(y_coords)
    center_x = []
    center_y = []

    for y in unique_y:
        # 获取当前 y 行上的所有 x 坐标
        xs_on_y = x_coords[y_coords == y]
        if len(xs_on_y) > 0:
            center_x.append(np.mean(xs_on_y))
            center_y.append(y)

    center_x = np.array(center_x)
    center_y = np.array(center_y)

    # 三次多项式拟合: x = a*y^3 + b*y^2 + c*y + d (注意：是以 y 为自变量预测 x，因为车道线通常是纵向的)
    poly_coeffs = np.polyfit(center_y, center_x, 3)

    return poly_coeffs, (center_x, center_y)


def main():
    detector = LaneDetectorTRT(ENGINE_PATH)
    print("🚀 TensorRT 模型加载成功！")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 无法打开 UVC 摄像头，请检查设备号或连接！")
        return

    print("开始处理视频流... 按 'q' 键退出。")
    fps_start = time.time()
    fps_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # 1. 模型推理获取 2D 掩码
        mask_2d = detector.infer(frame)

        # 2. 逆透视变换 (IPM) 得到鸟瞰图掩码
        bev_mask = cv2.warpPerspective(mask_2d, IPM_MATRIX, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        bev_display = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)  # 用于可视化的彩色 BEV 图

        # 3. 车道线中心提取与三次多项式拟合
        poly_coeffs, center_points = fit_lane_center(bev_mask)

        # 4. 可视化绘制
        # 绘制原始掩码到主图 (绿色)
        color_mask = np.zeros_like(frame)
        color_mask[mask_2d > 0] = [0, 255, 0]
        result_frame = cv2.addWeighted(frame, 0.8, color_mask, 0.4, 0)

        if poly_coeffs is not None:
            # 在 BEV 图上绘制拟合出的曲线 (红色)
            plot_y = np.linspace(0, bev_mask.shape[0] - 1, bev_mask.shape[0])
            plot_x = np.polyval(poly_coeffs, plot_y)

            # 将多项式点转为整数坐标用于 cv2.polylines
            pts = np.vstack((plot_x, plot_y)).astype(np.int32).T
            cv2.polylines(bev_display, [pts], isClosed=False, color=(0, 0, 255), thickness=5)

            # 选做：将拟合曲线投影回原图 (蓝色)
            pts_homo = np.hstack((pts, np.ones((pts.shape[0], 1))))  # [N, 3] 齐次坐标
            pts_orig = IPM_INV_MATRIX @ pts_homo.T  # 反向映射
            pts_orig = pts_orig / pts_orig[2, :]  # 归一化
            pts_orig = pts_orig[:2, :].T.astype(np.int32)

            # 过滤掉超出屏幕范围的点
            valid_pts = [p for p in pts_orig if 0 <= p[0] < frame.shape[1] and 0 <= p[1] < frame.shape[0]]
            if len(valid_pts) > 0:
                cv2.polylines(result_frame, [np.array(valid_pts)], isClosed=False, color=(255, 0, 0), thickness=3)

        # 性能统计
        t1 = time.time()
        fps_count += 1
        if t1 - fps_start > 1.0:
            print(f"FPS: {fps_count:.1f} | 推理及处理耗时: {(t1 - t0) * 1000:.1f}ms")
            fps_count = 0
            fps_start = t1

        # 显示
        cv2.putText(result_frame, "Blue: PolyFit Lane Center", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Lane Detection (Original View)", result_frame)
        cv2.imshow("Bird's Eye View (BEV)", bev_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()