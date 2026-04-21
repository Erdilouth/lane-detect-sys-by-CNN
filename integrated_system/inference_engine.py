"""
TensorRT推理引擎模块
Optimized TensorRT Inference Engine
"""

import cv2
import numpy as np
import tensorrt as trt
import logging
import os
from typing import Optional

try:
    from cuda import cuda, cudart
except ImportError:
    # 如果直接导入失败，尝试导入子模块
    import cuda.cuda as cuda
    import cuda.cudart as cudart

from .config import MODEL_CONFIG, get_model_path


class TensorRTInferenceEngine:
    """
    TensorRT推理引擎封装
    支持CUDA加速，优化内存管理
    """
    
    def __init__(self, engine_path: Optional[str] = None):
        """
        初始化TensorRT推理引擎
        
        Args:
            engine_path: TensorRT引擎文件路径，如果为None则使用配置文件中的路径
        """
        self.logger = logging.getLogger(__name__)
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        if engine_path is None:
            engine_path = get_model_path()
        
        self.engine_path = engine_path
        self.input_shape = MODEL_CONFIG["input_shape"]
        self.input_name = MODEL_CONFIG["input_name"]
        self.output_name = MODEL_CONFIG["output_name"]
        
        # 初始化CUDA
        self._init_cuda()
        
        # 加载引擎
        self._load_engine()
        
        # 分配内存
        self._allocate_memory()
        
        self.logger.info("✅ TensorRT推理引擎初始化成功")
    
    def _init_cuda(self):
        """初始化CUDA"""
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA初始化失败: {err}")
        self.logger.debug("CUDA初始化成功")
    
    def _load_engine(self):
        """加载TensorRT引擎"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {self.engine_path}")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape(self.input_name, self.input_shape)
        
        # 获取输出形状
        self.output_shape = self.context.get_tensor_shape(self.output_name)
        self.logger.debug(f"输出形状: {self.output_shape}")
    
    def _allocate_memory(self):
        """分配CUDA和主机内存"""
        # 计算所需内存大小（字节）
        self.input_size = trt.volume(self.input_shape) * 4  # float32 = 4 bytes
        self.output_size = trt.volume(self.output_shape) * 4
        
        # 分配CUDA设备内存
        err, self.d_input = cudart.cudaMalloc(self.input_size)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA输入内存分配失败: {err}")
        
        err, self.d_output = cudart.cudaMalloc(self.output_size)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA输出内存分配失败: {err}")
        
        # 分配主机内存
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        
        self.logger.debug("内存分配成功")
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            img: 输入图像 (BGR格式, HWC)
        
        Returns:
            预处理后的输入数据 (NCHW格式, float32, [0,1])
        """
        # 调整大小
        resized = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]并转换为float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        chw = normalized.transpose(2, 0, 1)
        
        # 添加batch维度: CHW -> NCHW
        nchw = np.expand_dims(chw, axis=0)
        
        # 确保内存连续
        return np.ascontiguousarray(nchw)
    
    def infer(self, img: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            img: 输入图像 (BGR格式, HWC)
        
        Returns:
            推理结果 (原始输出张量)
        """
        # 预处理
        input_data = self.preprocess(img)
        
        # Host -> Device
        cudart.cudaMemcpy(
            self.d_input,
            input_data.ctypes.data,
            self.input_size,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        
        # 设置输入输出地址
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        
        # 执行推理
        self.context.execute_v3(0)
        
        # Device -> Host
        cudart.cudaMemcpy(
            self.h_output.ctypes.data,
            self.d_output,
            self.output_size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        
        return self.h_output.copy()  # 返回副本以避免内存问题
    
    def infer_mask(self, img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        执行推理并返回二值化掩码
        
        Args:
            img: 输入图像 (BGR格式, HWC)
            threshold: 二值化阈值
        
        Returns:
            二值化掩码 (uint8, [0,255], 原始图像大小)
        """
        # 推理
        output = self.infer(img)
        
        # 提取掩码通道
        mask = output[0, 0]
        
        # 二值化
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # 恢复到原始图像大小
        mask_resized = cv2.resize(
            binary_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return mask_resized
    
    def cleanup(self):
        """释放资源"""
        if hasattr(self, 'd_input') and self.d_input:
            cudart.cudaFree(self.d_input)
        
        if hasattr(self, 'd_output') and self.d_output:
            cudart.cudaFree(self.d_output)
        
        self.logger.info("资源已释放")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass


# ============================================
# 单例模式（可选）
# ============================================
_engine_instance = None

def get_inference_engine(engine_path: Optional[str] = None) -> TensorRTInferenceEngine:
    """
    获取推理引擎单例
    
    Args:
        engine_path: 引擎路径（仅在首次创建时使用）
    
    Returns:
        TensorRTInferenceEngine实例
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TensorRTInferenceEngine(engine_path)
    return _engine_instance


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("测试TensorRT推理引擎...")
    
    try:
        engine = TensorRTInferenceEngine()
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # 测试推理
        print("执行推理测试...")
        output = engine.infer(test_img)
        print(f"输出形状: {output.shape}")
        
        # 测试掩码推理
        print("执行掩码推理测试...")
        mask = engine.infer_mask(test_img)
        print(f"掩码形状: {mask.shape}, 类型: {mask.dtype}")
        
        print("✅ 测试通过")
        
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()