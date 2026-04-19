"""
多线程帧缓冲区模块
Thread-Safe Frame Buffer for Multi-threaded Processing
"""

import cv2
import numpy as np
import logging
import threading
from queue import Queue, Empty, Full
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .config import THREAD_CONFIG


@dataclass
class FrameData:
    """
    帧数据结构
    """
    frame: np.ndarray          # 原始图像
    timestamp: float           # 时间戳
    frame_id: int              # 帧ID
    metadata: dict = None     # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FrameBuffer:
    """
    线程安全的帧缓冲区
    用于多线程环境下的帧数据传递
    """
    
    def __init__(self, max_size: int = None):
        """
        初始化帧缓冲区
        
        Args:
            max_size: 缓冲区最大大小
        """
        self.logger = logging.getLogger(__name__)
        
        if max_size is None:
            max_size = THREAD_CONFIG["buffer_size"]
        
        self.max_size = max_size
        
        # 使用队列实现
        self.buffer = Queue(maxsize=max_size)
        
        # 统计信息
        self.stats = {
            "total_put": 0,
            "total_get": 0,
            "dropped_frames": 0,
            "buffer_overflow": 0
        }
        
        # 锁（用于统计信息的线程安全访问）
        self.stats_lock = threading.Lock()
        
        self.logger.info(f"✅ 帧缓冲区初始化成功 (最大大小={max_size})")
    
    def put(self, frame: np.ndarray, 
            blocking: bool = False,
            metadata: dict = None) -> bool:
        """
        放入一帧数据
        
        Args:
            frame: 图像帧
            blocking: 是否阻塞等待（缓冲区满时）
            metadata: 元数据
        
        Returns:
            是否成功放入
        """
        frame_data = FrameData(
            frame=frame,
            timestamp=datetime.now().timestamp(),
            frame_id=self.stats["total_put"],
            metadata=metadata
        )
        
        try:
            if blocking:
                self.buffer.put(frame_data, block=True)
            else:
                self.buffer.put_nowait(frame_data)
            
            with self.stats_lock:
                self.stats["total_put"] += 1
            
            return True
            
        except Full:
            with self.stats_lock:
                self.stats["buffer_overflow"] += 1
            
            # 非阻塞模式下，尝试丢弃最旧的帧
            if not blocking:
                try:
                    self.buffer.get_nowait()
                    self.buffer.put_nowait(frame_data)
                    
                    with self.stats_lock:
                        self.stats["dropped_frames"] += 1
                        self.stats["total_put"] += 1
                    
                    self.logger.debug("缓冲区满，丢弃旧帧")
                    return True
                    
                except (Empty, Full):
                    return False
            
            return False
    
    def get(self, blocking: bool = False, 
            timeout: float = 0.1) -> Optional[FrameData]:
        """
        获取一帧数据
        
        Args:
            blocking: 是否阻塞等待
            timeout: 阻塞超时时间（秒）
        
        Returns:
            帧数据，失败返回None
        """
        try:
            if blocking:
                frame_data = self.buffer.get(block=True, timeout=timeout)
            else:
                frame_data = self.buffer.get_nowait()
            
            with self.stats_lock:
                self.stats["total_get"] += 1
            
            return frame_data
            
        except Empty:
            return None
    
    def peek(self) -> Optional[FrameData]:
        """
        查看最新一帧，但不移除
        
        Returns:
            最新帧数据，失败返回None
        """
        try:
            return self.buffer.queue[-1]
        except (Empty, IndexError):
            return None
    
    def clear(self):
        """清空缓冲区"""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
    
    def size(self) -> int:
        """获取当前缓冲区大小"""
        return self.buffer.qsize()
    
    def is_empty(self) -> bool:
        """判断缓冲区是否为空"""
        return self.buffer.empty()
    
    def is_full(self) -> bool:
        """判断缓冲区是否已满"""
        return self.buffer.full()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        with self.stats_lock:
            self.stats = {
                "total_put": 0,
                "total_get": 0,
                "dropped_frames": 0,
                "buffer_overflow": 0
            }


class SharedData:
    """
    共享数据容器
    用于线程间共享处理结果
    """
    
    def __init__(self):
        """初始化共享数据"""
        self.logger = logging.getLogger(__name__)
        
        # 共享数据
        self.data = {
            "latest_frame": None,
            "latest_mask": None,
            "latest_bev": None,
            "latest_coeffs": None,
            "processing_time": 0.0,
            "fps": 0.0
        }
        
        # 锁
        self.lock = threading.Lock()
        
        self.logger.info("✅ 共享数据容器初始化成功")
    
    def update(self, key: str, value: Any):
        """
        更新共享数据
        
        Args:
            key: 数据键
            value: 数据值
        """
        with self.lock:
            self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取共享数据
        
        Args:
            key: 数据键
            default: 默认值
        
        Returns:
            数据值
        """
        with self.lock:
            return self.data.get(key, default)
    
    def get_all(self) -> dict:
        """获取所有共享数据"""
        with self.lock:
            return self.data.copy()


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("测试帧缓冲区...")
    
    # 创建缓冲区
    buffer = FrameBuffer(max_size=5)
    
    # 测试放入帧
    print("测试放入帧...")
    for i in range(3):
        frame = np.random.zeros((100, 100, 3), dtype=np.uint8)
        success = buffer.put(frame, metadata={"index": i})
        print(f"放入帧 {i}: {'成功' if success else '失败'}")
    
    print(f"缓冲区大小: {buffer.size()}")
    
    # 测试获取帧
    print("\n测试获取帧...")
    for i in range(2):
        frame_data = buffer.get()
        if frame_data:
            print(f"获取帧 {frame_data.frame_id}, 元数据: {frame_data.metadata}")
        else:
            print("获取失败")
    
    print(f"缓冲区大小: {buffer.size()}")
    
    # 测试缓冲区满的情况
    print("\n测试缓冲区满的情况...")
    for i in range(10):
        frame = np.random.zeros((100, 100, 3), dtype=np.uint8)
        buffer.put(frame, blocking=False)
    
    stats = buffer.get_stats()
    print(f"统计信息: {stats}")
    
    # 测试共享数据
    print("\n测试共享数据...")
    shared = SharedData()
    shared.update("test_key", "test_value")
    value = shared.get("test_key")
    print(f"共享数据: {value}")
    
    print("✅ 测试通过")