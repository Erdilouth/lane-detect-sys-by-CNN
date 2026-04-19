"""
车道检测系统集成版
Integrated Lane Detection System

这是一个完整的车道检测系统，集成了TensorRT加速推理、
逆透视变换（IPM）、多项式车道线拟合和UART通信功能。
"""

__version__ = "1.0.0"
__author__ = "Integrated System Developer"

# 导出主要类和函数
from .config import (
    MODEL_CONFIG,
    CAMERA_CONFIG,
    IPM_CONFIG,
    LANE_FITTING_CONFIG,
    UART_CONFIG,
    THREAD_CONFIG,
    VISUALIZATION_CONFIG,
    PERFORMANCE_CONFIG,
    LOG_CONFIG,
    validate_config
)

from .inference_engine import (
    TensorRTInferenceEngine,
    get_inference_engine
)

from .ipm_transform import (
    IPMTransformer,
    get_ipm_transformer
)

from .lane_fitting import (
    LaneFitter,
    get_lane_fitter
)

from .uart_comm import (
    UARTCommunicator,
    get_uart_communicator
)

from .frame_buffer import (
    FrameBuffer,
    SharedData,
    FrameData
)

from .main_pipeline import (
    LaneDetectionPipeline
)

__all__ = [
    # 配置
    'MODEL_CONFIG',
    'CAMERA_CONFIG',
    'IPM_CONFIG',
    'LANE_FITTING_CONFIG',
    'UART_CONFIG',
    'THREAD_CONFIG',
    'VISUALIZATION_CONFIG',
    'PERFORMANCE_CONFIG',
    'LOG_CONFIG',
    'validate_config',
    
    # 推理引擎
    'TensorRTInferenceEngine',
    'get_inference_engine',
    
    # IPM变换
    'IPMTransformer',
    'get_ipm_transformer',
    
    # 车道线拟合
    'LaneFitter',
    'get_lane_fitter',
    
    # UART通信
    'UARTCommunicator',
    'get_uart_communicator',
    
    # 帧缓冲区
    'FrameBuffer',
    'SharedData',
    'FrameData',
    
    # 主管道
    'LaneDetectionPipeline',
]