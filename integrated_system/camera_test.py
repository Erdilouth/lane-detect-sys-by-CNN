#!/usr/bin/env python3
"""
CSI 摄像头测试脚本
用于在 Jetson Orin Nano 上测试 CSI 摄像头连接
"""

import sys

def test_gstreamer():
    """测试 GStreamer 是否可用"""
    try:
        import subprocess
        result = subprocess.run(
            ["which", "gst-launch-1.0"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ GStreamer 已安装")
            return True
        else:
            print("❌ GStreamer 未安装")
            print("请运行：sudo apt-get install gstreamer1.0-tools gstreamer1.0-nvidia")
            return False
    except Exception as e:
        print(f"❌ 检查 GStreamer 失败：{e}")
        return False

def test_camera_with_gstreamer(sensor_id=0, width=1280, height=720, fps=30):
    """使用 GStreamer 测试 CSI 摄像头"""
    try:
        import cv2
        
        pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink"
        )
        
        print(f"\n尝试使用 GStreamer 管道打开 CSI 摄像头 (sensor-id={sensor_id})...")
        print(f"管道：{pipeline}")
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"❌ 无法打开 sensor-id={sensor_id} 的摄像头")
            return False
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取帧")
            cap.release()
            return False
        
        print(f"✅ 摄像头打开成功！")
        print(f"   分辨率：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"   帧率：{int(cap.get(cv2.CAP_PROP_FPS))} FPS")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

def test_camera_simple(camera_id=0):
    """使用简单方式测试摄像头（USB 摄像头或 V4L2）"""
    try:
        import cv2
        
        print(f"\n尝试使用简单方式打开摄像头 (/dev/video{camera_id})...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取帧")
            cap.release()
            return False
        
        print(f"✅ 摄像头打开成功！")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Jetson Orin Nano CSI 摄像头测试")
    print("=" * 60)
    
    # 检查 GStreamer
    has_gstreamer = test_gstreamer()
    
    if has_gstreamer:
        # 尝试不同的 sensor-id
        for sensor_id in range(4):
            if test_camera_with_gstreamer(sensor_id=sensor_id):
                print(f"\n✅ 找到可用的 CSI 摄像头：sensor-id={sensor_id}")
                print("\n请在 config.py 中设置:")
                print('    CAMERA_CONFIG = {')
                print('        "camera_id": {},'.format(sensor_id))
                print('        "use_gstreamer": True,')
                print('        ...')
                print('    }')
                sys.exit(0)
    
    # 如果没有 GStreamer 或 CSI 摄像头失败，尝试简单方式
    print("\n尝试使用简单方式（适用于 USB 摄像头）...")
    for camera_id in range(4):
        if test_camera_simple(camera_id):
            print(f"\n✅ 找到可用的摄像头：/dev/video{camera_id}")
            print("\n请在 config.py 中设置:")
            print('    CAMERA_CONFIG = {')
            print('        "camera_id": {},'.format(camera_id))
            print('        "use_gstreamer": False,')
            print('        ...')
            print('    }')
            sys.exit(0)
    
    print("\n" + "=" * 60)
    print("❌ 未找到可用的摄像头")
    print("=" * 60)
    print("\n可能的原因:")
    print("1. CSI 摄像头未正确连接到 Jetson Orin Nano")
    print("2. 摄像头驱动程序未正确安装")
    print("3. 摄像头被其他程序占用")
    print("4. 需要使用 sudo 权限运行")
    print("\n调试命令:")
    print("  v4l2-ctl --list-devices")
    print("  ls -la /dev/video*")
    sys.exit(1)
