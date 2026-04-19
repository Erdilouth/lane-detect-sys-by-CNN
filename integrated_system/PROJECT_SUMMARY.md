# 车道检测系统集成版 - 项目总结
# Project Summary

## 📋 项目完成情况

✅ **所有任务已完成！**

本项目成功重写了原始代码，创建了一个完整的车道检测集成系统，实现了以下功能：

### 🎯 完成的功能

1. ✅ **TensorRT推理引擎** (`inference_engine.py`)
   - CUDA加速的深度学习推理
   - 优化的内存管理
   - 支持批量推理和掩码生成

2. ✅ **IPM逆透视变换** (`ipm_transform.py`)
   - 使用真实相机参数计算变换矩阵
   - 支持图像和坐标变换
   - 像素到米的转换

3. ✅ **车道线拟合** (`lane_fitting.py`)
   - 整合了原有的trajectory_fitter.py
   - 三次多项式拟合
   - 计算曲率和航向角

4. ✅ **UART通信** (`uart_comm.py`)
   - 优化了原有的uart_driver.py
   - 队列缓冲支持异步发送
   - 标准协议格式：`$LANE,a,b,c,d,*`

5. ✅ **多线程帧缓冲区** (`frame_buffer.py`)
   - 线程安全的队列实现
   - 智能丢帧策略
   - 共享数据容器

6. ✅ **多线程主程序** (`main_pipeline.py`)
   - 4个独立线程：采集、处理、UART、显示
   - 低延迟优化
   - 完整的错误处理

7. ✅ **统一配置系统** (`config.py`)
   - 集中管理所有配置参数
   - 配置验证功能
   - 路径工具函数

8. ✅ **工具函数库** (`utils.py`)
   - 多项式绘制
   - 曲率和航向角计算
   - 掩码叠加和可视化

## 📁 文件结构

```
integrated_system/
├── __init__.py           # 包初始化文件
├── config.py             # 统一配置
├── inference_engine.py   # TensorRT推理引擎
├── ipm_transform.py      # IPM变换模块
├── lane_fitting.py      # 车道线拟合模块
├── uart_comm.py          # UART通信模块
├── frame_buffer.py       # 多线程帧缓冲区
├── main_pipeline.py      # 主程序（多线程）
├── utils.py              # 工具函数
├── requirements.txt      # 依赖列表
├── README.md            # 使用文档
├── run.py               # 启动脚本
└── PROJECT_SUMMARY.md   # 本文档
```

## 🚀 使用方法

### 快速启动

```bash
# 进入集成系统目录
cd integrated_system

# 运行程序
python run.py
```

或者：

```bash
python main_pipeline.py
```

### 系统要求

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.19+
- TensorRT 8.0+
- CUDA 12.0+
- PySerial 3.5+

### 硬件要求

- NVIDIA GPU（支持CUDA）
- 摄像头（USB或Jetson相机）
- 串口设备（可选，用于UART通信）

## 🔧 配置说明

所有配置都在 `config.py` 中，主要配置项：

### 摄像头配置
- `camera_id`: 摄像头ID
- `use_gstreamer`: 是否使用GStreamer（Jetson相机）
- `width/height`: 图像分辨率
- `fps`: 目标帧率

### IPM配置
- `camera_params_path`: 相机参数文件路径
- `manual_src_points`: 手动指定ROI点
- `pixels_per_meter`: 像素到米的转换比例

### 车道线拟合配置
- `poly_order`: 多项式阶数（默认3）
- `min_points`: 最小点数阈值
- `start_x/end_x`: 采样距离范围

### UART配置
- `port`: 串口设备路径
- `baudrate`: 波特率
- `enable`: 是否启用通信

### 多线程配置
- `buffer_size`: 帧缓冲区大小
- `enable_display`: 是否启用显示
- `enable_uart`: 是否启用UART

## 📊 性能特性

### 多线程架构

系统采用4线程架构，实现真正的并行处理：

1. **采集线程**：独立运行，持续采集图像帧
2. **处理线程**：执行TensorRT推理和后处理
3. **UART线程**：异步发送多项式系数到下位机
4. **显示线程**：主程序，负责实时可视化

### 低延迟优化

- 非阻塞队列：避免线程阻塞
- 智能丢帧：缓冲区满时自动丢弃旧帧
- 内存预分配：CUDA内存预先分配，减少运行时开销
- 并行处理：推理和后处理可以并行执行

### 性能指标

预期性能：
- 推理延迟：~5-10ms（TensorRT）
- 总处理延迟：~20-30ms
- 系统FPS：30+ FPS
- UART延迟：<1ms（异步发送）

## 📡 UART协议

### 协议格式

```
$LANE,a,b,c,d,*\r\n
```

其中：
- `a, b, c, d`: 三次多项式系数
- 系数格式：保留6位小数
- 多项式：x = a*y³ + b*y² + c*y + d

### 示例输出

```
$LANE,0.000123,-0.123456,12.345678,456.789012,*
```

## 🧪 测试说明

每个模块都包含测试代码，可以独立测试：

```bash
# 测试配置
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

## 📝 日志系统

系统使用Python标准logging模块：

- 日志级别：DEBUG, INFO, WARNING, ERROR
- 日志格式：时间 - 模块名 - 级别 - 消息
- 可通过 `config.py` 中的 `LOG_CONFIG` 调整

## ⚠️ 注意事项

1. **TensorRT引擎文件**：确保 `mobilenet_lanenet.engine` 存在于项目根目录
2. **相机参数文件**：确保 `camera_params.json` 存在
3. **串口权限**：确保有访问 `/dev/ttyTHS0` 的权限
4. **CUDA环境**：确保CUDA和TensorRT正确安装
5. **摄像头连接**：确保摄像头正确连接并可访问

## 🐛 常见问题

### 摄像头无法打开

- 检查摄像头是否被其他程序占用
- 尝试更改 `camera_id`
- Jetson设备可以启用 `use_gstreamer`

### UART连接失败

```bash
# 检查设备权限
ls -l /dev/ttyTHS0
sudo chmod 666 /dev/ttyTHS0

# 或添加用户到dialout组
sudo usermod -a -G dialout $USER
```

### TensorRT引擎加载失败

```bash
# 检查引擎文件
ls -lh ../mobilenet_lanenet.engine

# 检查CUDA版本
nvcc --version

# 重新生成引擎
cd ..
python build_engine.py
```

## 🎨 可视化

系统提供两个可视化窗口：

1. **Lane Detection**（原始视角）
   - 绿色掩码：检测到的车道线区域
   - 蓝色曲线：拟合的车道线多项式
   - 信息文本：处理时间和FPS

2. **Bird's Eye View**（鸟瞰图）
   - 灰度掩码：IPM变换后的车道线
   - 红色曲线：拟合的多项式

## 🔍 代码改进说明

相比原始代码的主要改进：

1. **模块化设计**：功能分离，职责清晰
2. **多线程架构**：并行处理，降低延迟
3. **统一配置**：集中管理，易于调整
4. **错误处理**：完善的异常处理和日志
5. **性能优化**：内存预分配、非阻塞队列
6. **代码复用**：避免重复，提高可维护性
7. **文档完整**：详细的注释和使用说明

## 📚 参考资料

- 原始项目代码
- TensorRT文档
- OpenCV文档
- Python多线程编程

## 🙏 致谢

感谢原作者提供的车道检测框架基础代码，本集成系统在此基础上进行了重写和优化。

## 📄 许可证

本项目仅供学习和研究使用。

---

**开发完成时间**: 2026-04-19
**版本**: 1.0.0
**状态**: ✅ 完成并可使用