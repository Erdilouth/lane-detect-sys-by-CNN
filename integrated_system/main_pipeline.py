"""
主程序 - 多线程车道检测系统
Main Pipeline - Multi-threaded Lane Detection System
"""

import cv2
import numpy as np
import time
import logging
import threading
import sys
from typing import Optional

from .config import (
    CAMERA_CONFIG, THREAD_CONFIG, 
    VISUALIZATION_CONFIG, PERFORMANCE_CONFIG, LOG_CONFIG,
    validate_config
)
from .inference_engine import TensorRTInferenceEngine
from .ipm_transform import IPMTransformer
from .lane_fitting import LaneFitter
from .uart_comm import UARTCommunicator
from .frame_buffer import FrameBuffer, SharedData

# 尝试导入Libargus
try:
    import argus
    LIBARGUS_AVAILABLE = True
except ImportError:
    LIBARGUS_AVAILABLE = False


class LaneDetectionPipeline:
    """
    车道检测管道
    多线程处理：采集 ->、推理 -> 后处理 -> UART通信 -> 显示
    """
    
    def __init__(self):
        """初始化车道检测管道"""
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, LOG_CONFIG["level"]),
            format=LOG_CONFIG["format"]
        )
        self.logger = logging.getLogger(__name__)
        
        # 验证配置
        self.logger.info("验证配置...")
        try:
            validate_config()
        except ValueError as e:
            self.logger.error(f"配置验证失败: {e}")
            sys.exit(1)
        
        # 初始化各个模块
        self.logger.info("初始化模块...")
        self.inference_engine = TensorRTInferenceEngine()
        self.ipm_transformer = IPMTransformer()
        self.lane_fitter = LaneFitter()
        self.uart_comm = UARTCommunicator()
        
        # 初始化缓冲区和共享数据
        self.frame_buffer = FrameBuffer()
        self.shared_data = SharedData()
        
        # 控制标志
        self.running = False
        self.stop_event = threading.Event()
        
        # 摄像头
        self.cap = None
        self.camera = None  # Libargus相机对象
        self.stream = None  # Libargus流对象
        self.using_libargus = False  # 是否使用Libargus
        
        # 线程
        self.capture_thread = None
        self.process_thread = None
        self.uart_thread = None
        
        # 性能统计
        self.fps_count = 0
        self.fps_start = time.time()
        
        self.logger.info("✅ 车道检测管道初始化成功")
    
    def init_camera(self) -> bool:
        """
        初始化摄像头
        
        Returns:
            是否成功
        """
        use_libargus = CAMERA_CONFIG.get("use_libargus", False)
        
        if use_libargus:
            if not LIBARGUS_AVAILABLE:
                self.logger.error("❌ Libargus未安装，请先安装: sudo apt-get install python3-libargus")
                self.logger.info("提示: 回退到使用OpenCV直接调用摄像头")
                use_libargus = False
            else:
                return self._init_libargus_camera()
        
        # 使用OpenCV直接调用摄像头
        return self._init_opencv_camera()
    
    def _init_libargus_camera(self) -> bool:
        """使用Libargus初始化CSI摄像头"""
        try:
            import argus
            
            camera_id = CAMERA_CONFIG["camera_id"]
            width = CAMERA_CONFIG["width"]
            height = CAMERA_CONFIG["height"]
            fps = CAMERA_CONFIG["fps"]
            camera_mode = CAMERA_CONFIG.get("libargus_camera_mode", "1280x720")
            timeout = CAMERA_CONFIG.get("libargus_timeout", 10.0)
            
            self.logger.info(f"使用Libargus初始化CSI摄像头 (IMX219)")
            self.logger.info(f"   摄像头ID: {camera_id}")
            self.logger.info(f"   相机模式: {camera_mode}")
            self.logger.info(f"   目标分辨率: {width}x{height}")
            self.logger.info(f"   目标帧率: {fps} FPS")
            
            # 创建Argus相机
            self.camera = argus.Camera(camera_id)
            self.camera.open()
            
            # 创建流
            self.stream = argus.Stream(self.camera)
            self.stream.open()
            
            # 设置相机属性
            self.stream.set_mode(argus.PIXEL_FORMAT_YUV420, width, height)
            self.stream.set_fps(fps)
            
            # 创建帧捕获器
            self.stream.start_capture()
            
            self.using_libargus = True
            
            self.logger.info(f"✅ Libargus摄像头初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Libargus初始化失败: {e}")
            self.logger.info("提示: 回退到使用OpenCV直接调用摄像头")
            return False
    
    def _init_opencv_camera(self) -> bool:
        """使用OpenCV初始化USB摄像头"""
        camera_id = CAMERA_CONFIG.get("usb_camera_id", CAMERA_CONFIG["camera_id"])
        width = CAMERA_CONFIG["width"]
        height = CAMERA_CONFIG["height"]
        fps = CAMERA_CONFIG["fps"]
        preferred_fourcc = CAMERA_CONFIG.get("usb_camera_fourcc")

        # 尝试不同的后端打开摄像头
        backends = [
            cv2.CAP_ANY,
            cv2.CAP_V4L2,
        ]

        self.cap = None
        for backend in backends:
            self.logger.info(f"尝试使用后端 {backend} 打开USB摄像头: /dev/video{camera_id}")
            try:
                self.cap = cv2.VideoCapture(camera_id, backend)
                if self.cap.isOpened():
                    break
            except Exception as e:
                self.logger.warning(f"后端 {backend} 失败: {e}")
                self.cap = None
                continue

        if not self.cap or not self.cap.isOpened():
            self.logger.error("❌ 无法打开USB摄像头，请检查摄像头是否正确连接")
            self.logger.info(f"提示：可以尝试使用 'ls /dev/video*' 查看可用摄像头设备")
            return False

        # 首先尝试不设置编码格式，使用摄像头默认格式
        self.logger.info("使用摄像头默认格式初始化...")

        # 设置分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 对于某些摄像头，可能需要设置缓冲区大小
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 先测试读取一帧，看看是否正常
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            self.logger.warning("使用默认格式读取失败，尝试设置编码格式...")

            # 尝试不同的编码格式
            fourcc_options = []
            if preferred_fourcc:
                fourcc_options.append(preferred_fourcc)
            fourcc_options.extend(["YUYV", "MJPG", "YV12"])
            for fourcc in fourcc_options:
                self.logger.info(f"尝试编码格式: {fourcc}")
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

                # 再测试读取一帧
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    self.logger.info(f"编码格式 {fourcc} 可用")
                    break

        # 验证实际参数
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        # 解码fourcc
        actual_fourcc_str = "".join([
            chr((actual_fourcc >> 8 * i) & 0xFF)
            for i in range(4)
        ])

        self.logger.info(f"✅ USB摄像头初始化成功")
        self.logger.info(f"   配置分辨率: {width}x{height}")
        self.logger.info(f"   实际分辨率: {actual_width}x{actual_height}")
        self.logger.info(f"   目标帧率: {fps} FPS")
        self.logger.info(f"   实际帧率: {actual_fps} FPS")
        self.logger.info(f"   实际编码: {actual_fourcc_str}")

        # 再测试读取一帧并验证帧数据
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.logger.warning("⚠️  摄像头初始化成功但无法读取帧，尝试继续...")
        else:
            # 验证帧数据
            self.logger.info(f"帧形状: {test_frame.shape}, 数据类型: {test_frame.dtype}")
            if len(test_frame.shape) == 3:
                # 检查通道平均值，看看是否有异常绿色
                import numpy as np
                b_avg = np.mean(test_frame[:, :, 0])
                g_avg = np.mean(test_frame[:, :, 1])
                r_avg = np.mean(test_frame[:, :, 2])
                self.logger.debug(f"通道平均值 - R: {r_avg:.1f}, G: {g_avg:.1f}, B: {b_avg:.1f}")

                # 如果绿色通道异常高，可能需要检查颜色空间
                if g_avg > 200 and r_avg < 50 and b_avg < 50:
                    self.logger.warning("检测到异常绿色画面，可能需要检查颜色空间或编码格式")

        return True
    
    def capture_worker(self):
        """摄像头采集线程"""
        self.logger.info("📷 摄像头采集线程启动")
        
        if self.using_libargus:
            self._capture_libargus_frames()
        else:
            self._capture_opencv_frames()
        
        self.logger.info("📷 摄像头采集线程停止")
    
    def _capture_libargus_frames(self):
        """使用Libargus捕获帧"""
        try:
            import argus
            
            while not self.stop_event.is_set():
                # 从Libargus流中获取帧
                frame = self.stream.get_frame(timeout=10.0)
                
                if frame is None:
                    continue
                
                # 转换为OpenCV格式
                argus_frame = frame.get_array()
                # YUV420转BGR
                bgr_frame = cv2.cvtColor(argus_frame, cv2.COLOR_YUV2BGR_I420)
                
                # 放入缓冲区
                self.frame_buffer.put(bgr_frame, blocking=False)
                
        except Exception as e:
            self.logger.error(f"Libargus帧捕获错误: {e}")
    
    def _capture_opencv_frames(self):
        """使用OpenCV捕获USB摄像头帧"""
        consecutive_failures = 0
        max_consecutive_failures = 10

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures % 30 == 1:  # 避免频繁打印
                    self.logger.warning(f"USB摄像头读取失败 (连续失败: {consecutive_failures})")

                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error("USB摄像头连续读取失败过多，尝试重新初始化...")
                    self.cap.release()
                    if not self._init_opencv_camera():
                        self.logger.error("重新初始化USB摄像头失败，等待后重试...")
                        time.sleep(2)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0  # 重置失败计数

            # 检查帧数据的有效性，防止出现纯绿色画面
            import numpy as np
            valid_frame = True

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # 计算各通道平均值
                b_avg = np.mean(frame[:, :, 0])
                g_avg = np.mean(frame[:, :, 1])
                r_avg = np.mean(frame[:, :, 2])

                # 检查是否有异常的颜色值（如纯绿色）
                if g_avg > 230 and r_avg < 30 and b_avg < 30:
                    valid_frame = False
                    self.logger.warning("检测到纯绿色帧，丢弃...")

                # 检查颜色通道是否有严重失衡
                if abs(r_avg - g_avg) > 150 or abs(r_avg - b_avg) > 150 or abs(g_avg - b_avg) > 150:
                    valid_frame = False
                    self.logger.warning(f"颜色通道严重失衡 (R: {r_avg:.1f}, G: {g_avg:.1f}, B: {b_avg:.1f})，丢弃...")

            # 检查帧的亮度是否正常
            if valid_frame:
                avg_brightness = np.mean(frame)
                if avg_brightness < 5 or avg_brightness > 250:
                    valid_frame = False
                    self.logger.warning(f"帧亮度异常 ({avg_brightness:.1f})，丢弃...")

            # 如果帧有效，放入缓冲区
            if valid_frame:
                self.frame_buffer.put(frame, blocking=False)
    
    def process_worker(self):
        """处理线程（推理 + 后处理）"""
        self.logger.info("⚙️ 处理线程启动")
        
        while not self.stop_event.is_set():
            # 从缓冲区获取帧
            frame_data = self.frame_buffer.get(blocking=False)
            
            if frame_data is None:
                time.sleep(0.001)  # 避免CPU占用过高
                continue
            
            frame = frame_data.frame
            start_time = time.time()
            
            # 1. 推理
            try:
                mask = self.inference_engine.infer_mask(
                    frame, 
                    threshold=0.5
                )
                
                # 2. IPM变换
                bev_mask = self.ipm_transformer.transform(mask)
                
                # 3. 车道线拟合
                coeffs = self.lane_fitter.get_coeffs_for_uart(bev_mask)
                
                # 计算处理时间
                processing_time = time.time() - start_time
                
                # 更新共享数据
                self.shared_data.update("latest_frame", frame)
                self.shared_data.update("latest_mask", mask)
                self.shared_data.update("latest_bev", bev_mask)
                self.shared_data.update("latest_coeffs", coeffs)
                self.shared_data.update("processing_time", processing_time)
                
                # 4. 发送到UART队列
                if coeffs is not None and THREAD_CONFIG["enable_uart"]:
                    self.uart_comm.send_polynomial(coeffs, blocking=False)
                
                # 性能统计
                self.fps_count += 1
                
            except Exception as e:
                self.logger.error(f"处理错误: {e}")
        
        self.logger.info("⚙️ 处理线程停止")
    
    def uart_worker(self):
        """UART通信线程"""
        self.logger.info("📡 UART通信线程启动")
        
        while not self.stop_event.is_set():
            # 处理发送队列
            self.uart_comm.process_queue()
            time.sleep(0.01)  # 10ms轮询间隔
        
        self.logger.info("📡 UART通信线程停止")
    
    def visualize_frame(self, frame: np.ndarray, 
                        mask: np.ndarray,
                        bev_mask: np.ndarray,
                        coeffs: Optional[np.ndarray],
                        processing_time: float) -> np.ndarray:
        """
        可视化结果
        
        Args:
            frame: 原始帧
            mask: 掩码
            bev_mask: 鸟瞰图掩码
            coeffs: 多项式系数
            processing_time: 处理时间
        
        Returns:
            可视化图像
        """
        result = frame.copy()
        
        # 1. 绘制掩码
        if VISUALIZATION_CONFIG["show_mask"] and mask is not None:
            color_mask = np.zeros_like(frame)
            color_mask[mask > 0] = [0, 255, 0]
            result = cv2.addWeighted(
                result, 
                1 - VISUALIZATION_CONFIG["mask_alpha"],
                color_mask,
                VISUALIZATION_CONFIG["mask_alpha"],
                0
            )
        
        # 2. 绘制拟合曲线
        if VISUALIZATION_CONFIG["show_polyline"] and coeffs is not None:
            # 在鸟瞰图上绘制
            if bev_mask is not None:
                bev_display = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
                
                # 生成曲线点
                y_coords = np.linspace(0, bev_mask.shape[0] - 1, bev_mask.shape[0])
                x_coords = np.polyval(coeffs, y_coords)
                
                # 转换为整数坐标
                pts = np.vstack((x_coords, y_coords)).astype(np.int32).T
                
                # 过滤有效点
                valid_pts = [p for p in pts if 0 <= p[0] < bev_mask.shape[1]]
                
                if len(valid_pts) > 0:
                    cv2.polylines(
                        bev_display,
                        [np.array(valid_pts)],
                        isClosed=False,
                        color=(0, 0, 255),
                        thickness=VISUALIZATION_CONFIG["polyline_thickness"]
                    )
                
                # 更新共享数据
                self.shared_data.update("latest_bev_display", bev_display)
        
        # 3. 添加信息文本
        info_text = [
            f"Processing: {processing_time*1000:.1f}ms",
            f"FPS: {1.0/(processing_time+0.001):.1f}" if processing_time > 0 else "FPS: 0.0"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(
                result,
                text,
                (10, 30 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        return result
    
    def display_worker(self):
        """显示线程（主线程）"""
        self.logger.info("🖥️ 显示线程启动")
        
        fps_counter = 0
        fps_timer = time.time()
        
        while not self.stop_event.is_set():
            # 获取最新数据
            frame = self.shared_data.get("latest_frame")
            mask = self.shared_data.get("latest_mask")
            bev = self.shared_data.get("latest_bev")
            coeffs = self.shared_data.get("latest_coeffs")
            processing_time = self.shared_data.get("processing_time", 0.0)
            bev_display = self.shared_data.get("latest_bev_display")
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 可视化
            if THREAD_CONFIG["enable_display"]:
                result = self.visualize_frame(
                    frame, mask, bev, coeffs, processing_time
                )
                
                # 显示原图
                if VISUALIZATION_CONFIG["show_original"]:
                    cv2.imshow("Lane Detection", result)
                
                # 显示鸟瞰图
                if VISUALIZATION_CONFIG["show_bev"] and bev_display is not None:
                    cv2.imshow("Bird's Eye View", bev_display)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                
                # FPS统计
                fps_counter += 1
                if time.time() - fps_timer > 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    if PERFORMANCE_CONFIG["print_fps"]:
                        self.logger.info(f"显示FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_timer = time.time()
            else:
                time.sleep(0.01)
        
        self.logger.info("🖥️ 显示线程停止")
    
    def start(self):
        """启动管道"""
        self.logger.info("🚀 启动车道检测管道...")
        
        # 初始化摄像头
        if not self.init_camera():
            return False
        
        # 设置运行标志
        self.running = True
        self.stop_event.clear()
        
        # 启动采集线程
        self.capture_thread = threading.Thread(
            target=self.capture_worker,
            name="CaptureThread"
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self.process_worker,
            name="ProcessThread"
        )
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # 启动UART线程
        if THREAD_CONFIG["enable_uart"]:
            self.uart_thread = threading.Thread(
                target=self.uart_worker,
                name="UARTThread"
            )
            self.uart_thread.daemon = True
            self.uart_thread.start()
        
        self.logger.info("✅ 车道检测管道已启动")
        return True
    
    def run(self):
        """运行管道（主线程用于显示）"""
        if not self.start():
            return
        
        try:
            # 主线程运行显示
            self.display_worker()
            
        except KeyboardInterrupt:
            self.logger.info("接收到中断信号")
            self.stop_event.set()
        
        finally:
            self.stop()
    
    def stop(self):
        """停止管道"""
        self.logger.info("🛑 停止车道检测管道...")
        
        self.stop_event.set()
        self.running = False
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        if self.uart_thread and self.uart_thread.is_alive():
            self.uart_thread.join(timeout=2.0)
        
        # 释放Libargus资源
        if self.using_libargus:
            if self.stream:
                self.stream.stop_capture()
                self.stream.close()
            if self.camera:
                self.camera.close()
        
        # 释放OpenCV资源
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # 打印统计信息
        self.print_stats()
        
        self.logger.info("✅ 车道检测管道已停止")
    
    def print_stats(self):
        """打印统计信息"""
        self.logger.info("=" * 50)
        self.logger.info("统计信息:")
        
        # 帧缓冲区统计
        buffer_stats = self.frame_buffer.get_stats()
        self.logger.info(f"  帧缓冲区: {buffer_stats}")
        
        # UART统计
        uart_stats = self.uart_comm.get_stats()
        self.logger.info(f"  UART: {uart_stats}")
        
        self.logger.info("=" * 50)


def main():
    """主函数"""
    print("🚀 车道检测系统集成版")
    print("=" * 50)
    
    # 创建并运行管道
    pipeline = LaneDetectionPipeline()
    pipeline.run()
    
    print("程序退出")


if __name__ == "__main__":
    main()