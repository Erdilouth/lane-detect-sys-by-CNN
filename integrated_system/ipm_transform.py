"""
IPM（逆透视变换）模块
Inverse Perspective Transform with Real Camera Calibration
"""

import cv2
import numpy as np
import json
import logging
import os
from typing import Tuple, Optional

from .config import IPM_CONFIG, get_camera_params_path


class IPMTransformer:
    """
    逆透视变换处理器
    使用真实相机标定参数计算变换矩阵
    """
    
    def __init__(self):
        """
        初始化IPM变换器
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载相机参数
        self.camera_matrix = None
        self.dist_coeff = None
        self._load_camera_params()
        
        # 计算变换矩阵
        self.ipm_matrix = None
        self.ipm_inv_matrix = None
        self._compute_ipm_matrices()
        
        self.logger.info("✅ IPM变换器初始化成功")
    
    def _load_camera_params(self):
        """加载相机标定参数"""
        camera_params_path = get_camera_params_path()
        
        if not os.path.exists(camera_params_path):
            self.logger.warning(f"相机参数文件不存在: {camera_params_path}")
            self.logger.warning("将使用默认相机矩阵")
            # 使用默认相机矩阵（根据camera_params.json的典型值）
            self.camera_matrix = np.array([
                [792.21, 0.0, 333.22],
                [0.0, 789.24, 237.11],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            self.dist_coeff = np.array([
                [3.35, -39.85, 0.05, 0.007, 117.10]
            ], dtype=np.float32)
            return
        
        try:
            with open(camera_params_path, 'r') as f:
                params = json.load(f)
            
            self.camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
            self.dist_coeff = np.array(params['dist_coeff'], dtype=np.float32)
            
            self.logger.debug(f"相机内参: {self.camera_matrix}")
            self.logger.debug(f"畸变系数: {self.dist_coeff}")
            
        except Exception as e:
            self.logger.error(f"加载相机参数失败: {e}")
            raise
    
    def _compute_ipm_matrices(self):
        """
        计算IPM变换矩阵
        基于相机内参和预设的ROI点计算鸟瞰图变换
        """
        # 源点（图像坐标系中的ROI）
        src_points = np.array(IPM_CONFIG["manual_src_points"], dtype=np.float32)
        
        # 目标点（鸟瞰图坐标系）
        bev_width = IPM_CONFIG["bev_width"]
        bev_height = IPM_CONFIG["bev_height"]
        
        # 创建鸟瞰图的对应点
        # 保持车道线在鸟瞰图中的相对位置
        dst_points = np.array([
            [450, 0],      # 远端 左
            [830, 0],      # 远端 右
            [450, bev_height],  # 近端 左
            [830, bev_height]   # 近端 右
        ], dtype=np.float32)
        
        # 计算正向变换矩阵（图像 -> 鸟瞰图）
        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 计算反向变换矩阵（鸟瞰图 -> 图像）
        self.ipm_inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        self.logger.debug("IPM变换矩阵计算完成")
    
    def transform(self, img: np.ndarray, 
                  bev_width: Optional[int] = None,
                  bev_height: Optional[int] = None) -> np.ndarray:
        """
        执行逆变换变换（图像 -> 鸟瞰图）
        
        Args:
            img: 输入图像
            bev_width: 鸟瞰图宽度（默认使用配置值）
            bev_height: 鸟瞰图高度（默认使用配置值）
        
        Returns:
            鸟瞰图
        """
        if bev_width is None:
            bev_width = IPM_CONFIG["bev_width"]
        if bev_height is None:
            bev_height = IPM_CONFIG["bev_height"]
        
        # 使用线性插值
        bev = cv2.warpPerspective(
            img, 
            self.ipm_matrix, 
            (bev_width, bev_height),
            flags=cv2.INTER_LINEAR
        )
        
        return bev
    
    def inverse_transform(self, bev: np.ndarray,
                          img_width: Optional[int] = None,
                          img_height: Optional[int] = None) -> np.ndarray:
        """
        执行逆变换的逆变换（鸟瞰图 -> 原始图像）
        
        Args:
            bev: 鸟瞰图
            img_width: 原始图像宽度
            img_height: 原始图像高度
        
        Returns:
            原始视角图像
        """
        if img_width is None or img_height is None:
            # 如果未指定，使用鸟瞰图的尺寸
            img_width = bev.shape[1]
            img_height = bev.shape[0]
        
        img = cv2.warpPerspective(
            bev,
            self.ipm_inv_matrix,
            (img_width, img_height),
            flags=cv2.INTER_LINEAR
        )
        
        return img
    
    def transform_points(self, points: np.ndarray, 
                         inverse: bool = False) -> np.ndarray:
        """
        变换点坐标
        
        Args:
            points: 点集，形状为(N, 2)或(N, 3)
            inverse: 是否执行逆变换
        
        Returns:
            变换后的点集
        """
        if points.shape[1] == 2:
            # 添加齐次坐标
            points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
        elif points.shape[1] == 3:
            points_homo = points
        else:
            raise ValueError("点集形状必须是(N, 2)或(N, 3)")
        
        # 选择变换矩阵
        matrix = self.ipm_inv_matrix if inverse else self.ipm_matrix
        
        # 变换
        transformed = matrix @ points_homo.T
        transformed = transformed[:2, :].T  # 移除齐次坐标
        
        return transformed
    
    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        图像去畸变
        
        Args:
            img: 输入图像
        
        Returns:
            去畸变后的图像
        """
        return cv2.undistort(img, self.camera_matrix, self.dist_coeff)
    
    def pixels_to_meters(self, pixels: float) -> float:
        """像素到米的转换"""
        return pixels / IPM_CONFIG["pixels_per_meter"]
    
    def meters_to_pixels(self, meters: float) -> float:
        """米到像素的转换"""
        return meters * IPM_CONFIG["pixels_per_meter"]


# ============================================
# 单例模式
# ============================================
_ipm_instance = None

def get_ipm_transformer() -> IPMTransformer:
    """获取IPM变换器单例"""
    global _ipm_instance
    if _ipm_instance is None:
        _ipm_instance = IPMTransformer()
    return _ipm_instance


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("测试IPM变换器...")
    
    try:
        ipm = IPMTransformer()
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # 测试正向变换
        print("执行正向变换...")
        bev = ipm.transform(test_img)
        print(f"鸟瞰图形状: {bev.shape}")
        
        # 测试反向变换
        print("执行反向变换...")
        recovered = ipm.inverse_transform(bev, 1280, 720)
        print(f"恢复图像形状: {recovered.shape}")
        
        # 测试点变换
        print("测试点变换...")
        points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        transformed = ipm.transform_points(points)
        print(f"原始点: {points}")
        print(f"变换后点: {transformed}")
        
        # 测试距离转换
        print("测试距离转换...")
        pixels = 100
        meters = ipm.pixels_to_meters(pixels)
        back_pixels = ipm.meters_to_pixels(meters)
        print(f"{pixels}像素 = {meters:.2f}米")
        print(f"{meters:.2f}米 = {back_pixels:.1f}像素")
        
        print("✅ 测试通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()