# lane-detect-sys-by-CNN
# 基于CNN网络的车道检测与保持系统设计

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano-76B900.svg)
![TensorRT](https://img.shields.io/badge/Acceleration-TensorRT-green.svg)

## 📖 项目简介

本项目是为无人载货车辆设计的边缘侧自动驾驶核心子系统，专注于**车道保持 (Lane Keeping)** 功能。系统打通了从前端视觉感知、TensorRT 推理加速到后端 MPC+LQR 轨迹控制的完整技术闭环，旨在边缘计算设备上实现低延迟、高精度的车辆导航与控制。

*<img width="601" height="490" alt="系统框图" src="https://github.com/user-attachments/assets/6fc2f6be-c5c5-464a-bcfe-bc14db4e9069" />*

## ⚙️ 系统模块与完成情况

### 1. 边缘硬件部署 🟢 (已完成)
* **硬件平台**: 深度适配 Jetson Orin Nano 边缘计算盒子。
* **环境状态**: 已通过 WSL 配合 SDK Manager 成功刷入 JetPack 6.2，并完成底层运行环境的初始化。

### 2. 视觉感知与推理 🟢 (已完成)
* **模型架构**: 采用基于 MobileNet 构建的轻量级车道线检测网络。
* **部署与加速**: 已完成模型训练并成功导出 `mobilenet_lanenet.onnx`。通过 **TensorRT** 引擎实现了模型的极致推理加速，确保在嵌入式端的实时检测帧率。

### 3. 轨迹规划与控制 🟡 (未完成)
* **路线生成**: 对视觉感知模块输出的车道线进行多项式拟合，实时生成车辆的预期行驶轨迹。
* **核心控制器**: 基于 Python 独立实现了 **MPC (模型预测控制) + LQR (线性二次型调节器)** 组合控制算法，接收拟合轨迹并输出平滑的转向与速度控制指令。
* **可视化调试**: 使用 `matplotlib` 实现了控制曲线与轨迹的可视化，方便算法调参。
