# 车道检测系统集成版
# Integrated Lane Detection System

## 📖 项目简介

这是一个完整的车道检测系统，集成了TensorRT加速推理、逆透视变换（IPM）、多项式车道线拟合和UART通信功能。系统采用多线程架构，实现了低延迟的实时车道检测。

## ✨ 主要特性

- 🚀 **TensorRT加速推理**：使用CUDA实现高性能深度学习推理
- 📷 **实时图像处理**：支持USB摄像头和Jetson相机（GStreamer）
- 🔄 **逆透视变换（IPM）**：基于真实相机参数计算鸟瞰图
- 📐 **多项式拟合**：三次多项式拟合车道线，计算曲率和航向角
- 📡 **UART通信**：通过串口向下位机发送多项式系数
- 🧵 **多线程架构**：采集、推理、处理、通信并行执行
- ⚡ **低延迟优化**：使用线程安全队列和智能丢帧策略
- 🖥️ **实时可视化**：同时显示原始视图和鸟瞰图

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     主线程 (显示)                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  可视化显示   │←→│  共享数据    │←→│  帧缓冲数据   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ↑                 ↑                 ↑
         │                 │                 │
┌────────┴─────────────────┴─────────────────┴────────┐
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  采集线程    │  │  处理线程    │  │  UART线程    │ │
│  │  (摄像头)    │  │  (推理+拟合) │  │  (串口通信)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  模块：                                                  │
│  • TensorRT推理引擎                                       │
│  • IPM变换器                                             │
│  • 车道线拟合器                                          │
│  • UART通信器                                             │
│                                                          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
integrated_system/
├── config.py              # 统一配置文件
├── inference_engine.py    # TensorRT推理引擎
├── ipm_transform.py       # 逆透视变换模块
├── lane_fitting.py       # 车道线拟合模块
├── uart_comm.py           # UART通信模块
├── frame_buffer.py        # 多线程帧缓冲区
├── main_pipeline.py       # 主程序（多线程处理）
├── utils.py               # 工具函数
├── requirements.txt       # 依赖列表
└── README.md             # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保TensorRT引擎文件存在
# mobilenet_lanenet.engine 应该在项目根目录
```

### 2. 配置文件

编辑 `config.py` 根据你的硬件和环境调整配置：

```python
# 摄像头配置
CAMERA_CONFIG = {
    "camera_id": 0,          # USB摄像头ID
    "use_gstreamer": False,  # 是否使用GStreamer（Jetson相机）
    # ...
}

# UART配置
UART_CONFIG = {
    "port": "/dev/ttyTHS0",  # 串口设备
    "enable": True,            # 是否启用UART
    # ...
}
```

### 3. 运行程序

```bash
# 在integrated_system目录下运行
python main_pipeline.py
```

### 4. 控制说明

- 按 `q` 键退出程序
- 程序会显示两个窗口：
  - `Lane Detection`: 原始视角检测结果
  - `Bird's Eye View`: 鸟瞰图视角

## 🔧 配置说明

### 摄像头配置

```python
CAMERA_CONFIG = {
    "camera_id": 0,              # 摄像头ID
    "width": 1280,              # 图像宽度
    "height": 720,              # 图像高度
    "fps": 30,                 # 目标帧率
    "use_gstreamer": False,      # 是否使用GStreamer
}
```

### IPM配置

```python
IPM_CONFIG = {
    "camera_params_path": "../camera_params.json",  # 相机参数文件
    "manual_src_points": [...],  # ROI区域（可选）
    "bev_width": 1280,          # 鸟瞰图宽度
    "bev_height": 720,          # 鸟瞰图高度
    "pixels_per_meter": 40.0,    # 像素到米的转换比例
}
```

### 车道线拟合配置

```python
LANE_FITTING_CONFIG = {
    "poly_order": 3,      # 多项式阶数
    "min_points": 50,     # 最小点数阈值
    "start_x": 0.0,      # 采样起始距离（米）
    "end_x": 15.0,       # 采样终止距离（米）
    "step": 0.1,         # 采样步长（米）
}
```

### UART配置

```python
UART_CONFIG = {
    "port": "/dev/ttyTHS0",  # 串口设备路径
    "baudrate": 115200,       # 波特率
    "timeout": 0.1,           # 超时时间
    "enable": True,            # 是否启用通信
}
```

### 多线程配置

```python
THREAD_CONFIG = {
    "buffer_size": 5,        # 帧缓冲区大小
    "enable_display": True,   # 是否启用显示
    "enable_uart": True,      # 是否启用UART
    "max_fps": 30,          # 最大FPS限制
}
```

## 📊 UART协议

系统通过UART向下位机发送车道线多项式系数：

### 协议格式

```
$LANE,a,b,c,d,*\r\n
```

其中：
- `a, b, c, d`: 三次多项式系数 (x = a*y³ + b*y² + c*y + d)
- 系数保留6位小数

### 示例

```
$LANE,0.000100,-0.100000,10.000000,400.000000,*
```

## 🔍 性能优化

### 1. 多线程架构

- **采集线程**：独立运行，持续采集图像
- **处理线程**：执行推理和后处理
- **UART线程**：异步发送数据
- **显示线程**：主线程，负责可视化

### 2. 低延迟策略

- 使用非阻塞队列传递数据
- 缓冲区满时自动丢弃旧帧
- 智能丢帧确保实时性

### 3. 内存优化

- CUDA内存预分配
- 复用缓冲区
- 避免不必要的内存拷贝

## 🧪 测试各个模块

```bash
# 测试配置文件
python config.py

# 测试推理引擎
python inference_engine.py

# 测试IPM变换
python ipm_transform.py

# 测试车道线拟合
python lane_fitting.py

# 测试UART通信
python uart_comm.py

# 测试帧缓冲区
python frame_buffer.py

# 测试工具函数
python utils.py
```

## 📝 日志说明

系统使用Python标准logging模块，日志级别可通过配置调整：

```python
LOG_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
}
```

日志输出示例：

```
2026-04-19 23:00:00 - inference_engine - INFO - ✅ TensorRT推理引擎初始化成功
2026-04-19 23:00:00 - ipm_transform - INFO - ✅ IPM变换器初始化成功
2026-04-19 23:00:00 - lane_fitting - INFO - ✅ 车道线拟合器初始化成功 (阶数=3)
2026-04-19 23:00:00 - uart_comm - INFO - ✅ UART连接成功: /dev/ttyTHS0 @ 115200 bps
2026-04-19 23:00:00 - main_pipeline - INFO - 🚀 启动车道检测管道...
```

## ⚠️ 注意事项

1. **TensorRT引擎**：确保 `mobilenet_lanenet.engine` 文件存在
2. **相机标定**：使用 `camera_params.json` 中的真实参数
3. **串口权限**：确保有访问串口设备的权限
4. **CUDA支持**：确保NVIDIA驱动和CUDA正确安装
5. **Jetson设备**：如果使用Jetson相机，需要配置GStreamer

## 🐛 故障排除

### 摄像头无法打开

```python
# 检查摄像头是否被占用
# 尝试更改camera_id
# 或者启用use_gstreamer（Jetson设备）
```

### UART连接失败

```bash
# 检查串口设备是否存在
ls -l /dev/ttyTHS0

# 检查权限
sudo chmod 666 /dev/ttyTHS0

# 或将用户加入dialout组
sudo usermod -a -G dialout $USER
```

### TensorRT引擎加载失败

```bash
# 检查引擎文件是否存在
ls -lh mobilenet_lanenet.engine

# 检查CUDA版本
nvcc --version

# 重新生成引擎文件
python build_engine.py
```

## 📄 许可证

本项目仅供学习和研究使用。

## 👨‍💻 开发者

- 集成系统版本
- 基于原有模块重写和优化

## 🙏 致谢

感谢原作者提供的车道检测框架基础代码。