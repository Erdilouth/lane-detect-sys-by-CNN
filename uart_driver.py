import serial
import time
import logging

class UARTDriver:
    def __init__(self, port='/dev/ttyTHS0', baudrate=115200, timeout=0.1):
        """
        初始化 Jetson Orin Nano 的串口通信
        注意：Jetson 40-pin 上的 UART 通常映射为 /dev/ttyTHS0 或 /dev/ttyTHS1
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connect()

    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logging.info(f"✅ UART 成功连接: {self.port} @ {self.baudrate} bps.")
        except serial.SerialException as e:
            logging.error(f"❌ UART 连接失败 ({self.port}): {e}")

    def send_polynomial(self, coeffs):
        """
        向下位机发送车道线多项式系数。
        coeffs: 按照 [a, b, c, d] 顺序排列的三阶多项式系数数组
        """
        if self.serial_conn and self.serial_conn.is_open:
            if len(coeffs) == 4:
                a, b, c, d = coeffs
                # 构造符合我们定义的协议帧
                cmd_str = f"$LANE,{a:.6f},{b:.6f},{c:.6f},{d:.6f},*\r\n"
                try:
                    self.serial_conn.write(cmd_str.encode('ascii'))
                    # logging.debug(f"已发送: {cmd_str.strip()}")
                except Exception as e:
                    logging.error(f"UART 写入错误: {e}")
            else:
                logging.warning("多项式系数长度不为4，跳过发送。")

    def close(self):
        if self.serial_conn:
            self.serial_conn.close()
            logging.info("UART 连接已关闭。")