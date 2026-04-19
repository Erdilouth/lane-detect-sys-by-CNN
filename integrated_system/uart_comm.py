"""
UART通信模块
UART Communication Module - Optimized from uart_driver.py
"""

import serial
import logging
import threading
from typing import Optional, List
from queue import Queue, Empty

from .config import UART_CONFIG


class UARTCommunicator:
    """
    UART通信器
    用于向下位机发送车道线多项式系数
    """
    
    def __init__(self, port: str = None, baudrate: int = None, 
                 timeout: float = None, enable: bool = None):
        """
        初始化UART通信器
        
        Args:
            port: 串口设备路径
            baudrate: 波特率
            timeout: 超时时间（秒）
            enable: 是否启用通信
        """
        self.logger = logging.getLogger(__name__)
        
        # 从配置获取参数
        if port is None:
            port = UART_CONFIG["port"]
        if baudrate is None:
            baudrate = UART_CONFIG["baudrate"]
        if timeout is None:
            timeout = UART_CONFIG["timeout"]
        if enable is None:
            enable = UART_CONFIG["enable"]
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.enable = enable
        
        # 串口连接
        self.serial_conn = None
        
        # 发送队列（线程安全）
        self.send_queue = Queue(maxsize=100)
        
        # 统计信息
        self.stats = {
            "sent_count": 0,
            "error_count": 0,
            "queue_overflow": 0
        }
        
        # 连接串口
        if self.enable:
            self.connect()
        
        self.logger.info("✅ UART通信器初始化成功")
    
    def connect(self) -> bool:
        """
        连接串口
        
        Returns:
            连接是否成功
        """
        if not self.enable:
            self.logger.warning("UART通信已禁用")
            return False
        
        try:
            self.serial_conn = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            self.logger.info(f"✅ UART连接成功: {self.port} @ {self.baudrate} bps")
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"❌ UART连接失败 ({self.port}): {e}")
            self.serial_conn = None
            return False
    
    def reconnect(self) -> bool:
        """
        重新连接串口
        """
        self.close()
        return self.connect()
    
    def send_polynomial(self, coeffs: List[float], blocking: bool = False) -> bool:
        """
        发送多项式系数（使用队列）
        
        Args:
            coeffs: 多项式系数 [a, b, c, d]
            blocking: 是否阻塞等待（队列满时）
        
        Returns:
            是否成功加入队列
        """
        if not self.enable:
            return False
        
        if not self._validate_coeffs(coeffs):
            self.logger.warning(f"无效的多项式系数: {coeffs}")
            return False
        
        try:
            if blocking:
                self.send_queue.put(coeffs, block=True)
            else:
                self.send_queue.put_nowait(coeffs)
            return True
            
        except Exception as e:
            self.stats["queue_overflow"] += 1
            self.logger.error(f"发送队列已满: {e}")
            return False
    
    def send_polynomial_direct(self, coeffs: List[float]) -> bool:
        """
        直接发送多项式系数（不使用队列）
        
        Args:
            coeffs: 多项式系数 [a, b, c, d]
        
        Returns:
            是否发送成功
        """
        if not self.enable:
            return False
        
        if not self.serial_conn or not self.serial_conn.is_open:
            self.logger.warning("UART未连接，尝试重新连接...")
            if not self.reconnect():
                return False
        
        if not self._validate_coeffs(coeffs):
            return False
        
        try:
            # 构造协议帧: $LANE,a,b,c,d,*\r\n
            cmd_str = self._format_command(coeffs)
            
            # 发送
            self.serial_conn.write(cmd_str.encode('ascii'))
            self.serial_conn.flush()
            
            self.stats["sent_count"] += 1
            self.logger.debug(f"发送成功: {cmd_str.strip()}")
            return True
            
        except Exception as e:
            self.stats["error_count"] += 1
            self.logger.error(f"UART发送错误: {e}")
            return False
    
    def process_queue(self) -> int:
        """
        处理发送队列中的所有消息
        
        Returns:
            成功发送的消息数量
        """
        if not self.enable:
            return 0
        
        sent_count = 0
        
        while True:
            try:
                # 非阻塞获取
                coeffs = self.send_queue.get_nowait()
                
                # 发送
                if self.send_polynomial_direct(coeffs):
                    sent_count += 1
                    
            except Empty:
                # 队列为空，退出
                break
            except Exception as e:
                self.logger.error(f"处理队列错误: {e}")
                break
        
        return sent_count
    
    def _validate_coeffs(self, coeffs: List[float]) -> bool:
        """验证多项式系数"""
        return coeffs is not None and len(coeffs) == 4
    
    def _format_command(self, coeffs: List[float]) -> str:
        """格式化命令字符串"""
        a, b, c, d = coeffs
        return f"$LANE,{a:.6f},{b:.6f},{c:.6f},{d:.6f},*\r\n"
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "sent_count": 0,
            "error_count": 0,
            "queue_overflow": 0
        }
    
    def close(self):
        """关闭串口连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.logger.info("UART连接已关闭")
        
        # 清空队列
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
            except Empty:
                break
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass


# ============================================
# 单例模式
# ============================================
_uart_instance = None

def get_uart_communicator(port: str = None, 
                           baudrate: int = None,
                           timeout: float = None,
                           enable: bool = None) -> UARTCommunicator:
    """获取UART通信器单例"""
    global _uart_instance
    if _uart_instance is None:
        _uart_instance = UARTCommunicator(port, baudrate, timeout, enable)
    return _uart_instance


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("测试UART通信器...")
    
    # 创建通信器（不启用实际连接）
    comm = UARTCommunicator(enable=False)
    
    # 测试系数验证
    print("测试系数验证...")
    valid_coeffs = [0.0001, -0.1, 10.0, 400.0]
    print(f"有效系数: {valid_coeffs}")
    print(f"验证结果: {comm._validate_coeffs(valid_coeffs)}")
    
    # 测试命令格式化
    print("测试命令格式化...")
    cmd_str = comm._format_command(valid_coeffs)
    print(f"命令字符串: {cmd_str.strip()}")
    
    # 测试队列发送
    print("测试队列发送...")
    for i in range(5):
        coeffs = [0.0001 + i*0.00001, -0.1 + i*0.01, 10.0 + i, 400.0 + i*10]
        comm.send_polynomial(coeffs, blocking=False)
    
    print(f"队列大小: {comm.send_queue.qsize()}")
    
    # 测试统计信息
    print("测试统计信息...")
    stats = comm.get_stats()
    print(f"统计信息: {stats}")
    
    print("✅ 测试通过")
    
    comm.close()