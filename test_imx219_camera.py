#!/usr/bin/env python3
"""
测试IMX219 CSI摄像头配置
Test IMX219 CSI Camera Configuration
"""

import cv2
import time
import sys

def test_gstreamer_pipeline():
    """测试GStreamer管道"""
    
    # IMX219 CSI摄像头配置
    sensor_id = 0
    width = 1280
    height = 720
    fps = 30
    
    # 构建GStreamer管道
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink"
    )
    
    print("=" * 60)
    print("IMX219 CSI摄像头测试")
    print("=" * 60)
    print(f"GStreamer管道:")
    print(pipeline)
    print("=" * 60)
    
    # 尝试打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("\n可能的原因:")
        print("1. IMX219摄像头未正确连接")
        print("2. nvarguscamerasrc插件未安装或配置错误")
        print("3. Jetson ISP未正确配置")
        print("4. 传感器ID不正确")
        print("\n调试命令:")
        print("  v4l2-ctl --list-devices")
        print("  gst-launch-1.0 nvarguscamerasrc ! fakesink")
        return False
    
    # 获取实际参数
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ 摄像头打开成功")
    print(f"   分辨率: {actual_width}x{actual_height}")
    print(f"   帧率: {actual_fps} FPS")
    print("\n开始捕获画面（按'q'退出）...")
    
    # 测试帧捕获
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ 无法读取帧")
                break
            
            frame_count += 1
            
            # 显示画面
            cv2.imshow("IMX219 Camera Test", frame)
            
            # 计算FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f} | 帧数: {frame_count}")
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印统计
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print("\n" + "=" * 60)
        print("测试结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {elapsed:.2f}秒")
        print(f"  平均FPS: {avg_fps:.1f}")
        print("=" * 60)
    
    return True

def test_usb_camera():
    """测试USB摄像头（备用方案）"""
    
    print("\n" + "=" * 60)
    print("测试USB摄像头（备用方案）")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开USB摄像头")
        return False
    
    print("✅ USB摄像头打开成功")
    print("正在捕获画面（按'q'退出）...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("USB Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    # 首先尝试GStreamer（IMX219）
    success = test_gstreamer_pipeline()
    
    if not success:
        print("\nGStreamer测试失败，是否尝试USB摄像头？")
        print("按Enter继续USB摄像头测试，或Ctrl+C退出")
        try:
            input()
            test_usb_camera()
        except KeyboardInterrupt:
            print("\n退出")
            sys.exit(0)