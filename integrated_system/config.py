"""
车道检测系统集成配置文件
Configuration for Integrated Lane Detection System
"""

import os

# ============================================
# 模型配置
# ============================================
MODEL_CONFIG = {
    "engine_path": "/home/erdilouth/mobilenet_lanenet.engine",  # TensorRT引擎路径
    "input_shape": (1, 3, 256, 512),  # (Batch, Channel, Height, Width)
    "input_name": "input",
    "output_name": "output",
}

# ============================================
# 摄像头配置
# ============================================
CAMERA_CONFIG = {
    "camera_id": 0,  # 摄像头ID（USB摄像头通常为0）
    "width": 1280,   # 图像宽度
    "height": 720,   # 图像高度
    "fps": 30,       # 目标帧率
    
    # Jetson CSI摄像头配置（IMX219）
    "use_gstreamer": True,  # 启用GStreamer以使用CSI摄像头
    "sensor_id": 0,         # IMX219传感器ID（通常为0）
    "sensor_mode": 3,       # 传感器模式：3=1280x720 30fps
    "gstreamer_pipeline": (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink"
    )
}

# ============================================
# IPM（逆透视变换）配置
# ============================================
IPM_CONFIG = {
    # 使用真实相机标定参数
    "camera_params_path": "camera_params.json",
    
    # 手动指定ROI区域（如果自动计算失败，使用这些值）
    "manual_src_points": [
        [450, 460], [830, 460],  # 远端 左、右
        [150, 720], [1130, 720]   # 近端 左、右
    ],
    
    # 鸟瞰图尺寸
    "bev_width": 1280,
    "bev_height": 720,
    
    # 物理尺寸映射（像素到米的转换）
    "pixels_per_meter": 40.0,  # 每米对应的像素数
}

# ============================================
# 车道线拟合配置
# ============================================
LANE_FITTING_CONFIG = {
    "poly_order": 3,           # 多项式阶数（3次）
    "min_points": 50,          # 最小点数（点数太少则不拟合）
    "start_x": 0.0,            # 采样起始距离（米）
    "end_x": 15.0,             # 采样终止距离（米）
    "step": 0.1,               # 采样步长（米）
    "threshold": 0.5,          # 掩码阈值
}

# ============================================
# UART通信配置
# ============================================
UART_CONFIG = {
    "port": "/dev/ttyTHS0",    # Jetson串口设备
    "baudrate": 115200,        # 波特率
    "timeout": 0.1,            # 超时时间（秒）
    "enable": True,            # 是否启用UART通信
}

# ============================================
# 多线程配置
# ============================================
THREAD_CONFIG = {
    "buffer_size": 5,          # 帧缓冲区大小
    "enable_display": True,    # 是否启用显示
    "enable_uart": True,       # 是否启用UART发送
    "max_fps": 30,             # 最大FPS限制
}

# ============================================
# 日志配置
# ============================================
LOG_CONFIG = {
    "level": "INFO",           # 日志级别: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# ============================================
# 可视化配置
# ============================================
VISUALIZATION_CONFIG = {
    "show_original": True,     # 显示原始图像
    "show_mask": True,         # 显示掩码
    "show_bev": True,          # 显示鸟瞰图
    "show_polyline": True,     # 显示拟合的曲线
    "mask_alpha": 0.4,         # 掩码透明度
    "polyline_thickness": 5,   # 多项式曲线粗细
}

# ============================================
# 性能监控
# ============================================
PERFORMANCE_CONFIG = {
    "enable_profiling": True,  # 启用性能分析
    "print_fps": True,         # 打印FPS信息
    "print_latency": True,     # 打印各模块延迟
}

# ============================================
# 路径工具函数
# ============================================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path():
    """获取模型路径"""
    return os.path.join(get_project_root(), MODEL_CONFIG["engine_path"])

def get_camera_params_path():
    """获取相机参数路径"""
    return os.path.join(get_project_root(), IPM_CONFIG["camera_params_path"])

# ============================================
# 验证配置
# ============================================
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查模型文件是否存在
    if not os.path.exists(get_model_path()):
        errors.append(f"模型文件不存在: {get_model_path()}")
    
    # 检查相机参数文件是否存在
    if not os.path.exists(get_camera_params_path()):
        errors.append(f"相机参数文件不存在: {get_camera_params_path()}")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    return True

if __name__ == "__main__":
    # 测试配置
    print("配置文件测试...")
    print(f"项目根目录: {get_project_root()}")
    print(f"模型路径: {get_model_path()}")
    print(f"相机参数路径: {get_camera_params_path()}")
    
    try:
        validate_config()
        print("✅ 配置验证通过")
    except ValueError as e:
        print(f"❌ {e}")