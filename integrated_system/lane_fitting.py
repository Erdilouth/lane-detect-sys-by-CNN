"""
车道线拟合模块
Lane Fitting Module - Integrated from trajectory_fitter.py
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .config import LANE_FITTING_CONFIG


class LaneFitter:
    """
    车道线拟合器
    从鸟瞰图掩提取车道线并进行多项式拟合
    """
    
    def __init__(self, poly_order: int = None):
        """
        初始化车道线拟合器
        
        Args:
            poly_order: 多项式阶数，默认使用配置值
        """
        self.logger = logging.getLogger(__name__)
        
        if poly_order is None:
            self.poly_order = LANE_FITTING_CONFIG["poly_order"]
        else:
            self.poly_order = poly_order
        
        self.min_points = LANE_FITTING_CONFIG["min_points"]
        self.threshold = LANE_FITTING_CONFIG["threshold"]
        
        self.logger.info(f"✅ 车道线拟合器初始化成功 (阶数={self.poly_order})")
    
    def extract_center_points(self, bev_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从鸟瞰图掩码中提取车道线中心点
        
        Args:
            bev_mask: 鸟瞰图二值掩码 (0或255)
        
        Returns:
            (center_x, center_y): 中心点坐标数组
        """
        # 找到所有车道线像素点的坐标
        y_coords, x_coords = np.where(bev_mask > self.threshold)
        
        if len(y_coords) < self.min_points:
            self.logger.warning(f"检测到的点数太少: {len(y_coords)} < {self.min_points}")
            return np.array([]), np.array([])
        
        # 计算每一行 (y) 的中心点 (x_mean)
        unique_y = np.unique(y_coords)
        center_x = []
        center_y = []
        
        for y in unique_y:
            # 获取当前 y 行上的所有 x 坐标
            xs_on_y = x_coords[y_coords == y]
            if len(xs_on_y) > 0:
                center_x.append(np.mean(xs_on_y))
                center_y.append(y)
        
        center_x = np.array(center_x)
        center_y = np.array(center_y)
        
        return center_x, center_y
    
    def fit_polynomial(self, center_x: np.ndarray, 
                       center_y: np.ndarray) -> Optional[np.ndarray]:
        """
        对中心点进行多项式拟合
        
        Args:
            center_x: x坐标数组
            center_y: y坐标数组
        
        Returns:
            多项式系数 (从高次到低次)，失败返回None
        """
        if len(center_x) < self.poly_order + 1:
            self.logger.warning(f"点数不足以拟合{self.poly_order}阶多项式")
            return None
        
        try:
            # 多项式拟合: x = a*y^3 + b*y^2 + c*y + d
            # 以 y 为自变量预测 x，因为车道线通常是纵向的
            coeffs = np.polyfit(center_y, center_x, self.poly_order)
            return coeffs
        except Exception as e:
            self.logger.error(f"多项式拟合失败: {e}")
            return None
    
    def fit_and_sample(self, 
                       bev_mask: np.ndarray,
                       start_x: float = None,
                       end_x: float = None,
                       step: float = None) -> Optional[Dict[str, np.ndarray]]:
        """
        完整的拟合和采样流程
        
        Args:
            bev_mask: 鸟瞰图掩码
            start_x: 采样起始距离（米）
            end_x: 采样终止距离（米）
            step: 采样步长（米）
        
        Returns:
            包含拟合结果的字典:
            {
                "coeffs": 多项式系数,
                "center_x": 中心点x坐标,
                "center_y": 中心点y坐标,
                "sampled_x": 采样点x坐标,
                "sampled_y": 采样点y坐标,
                "yaw": 航向角,
                "curvature": 曲率
            }
        """
        if start_x is None:
            start_x = LANE_FITTING_CONFIG["start_x"]
        if end_x is None:
            end_x = LANE_FITTING_CONFIG["end_x"]
        if step is None:
            step = LANE_FITTING_CONFIG["step"]
        
        # 1. 提取中心点
        center_x, center_y = self.extract_center_points(bev_mask)
        
        if len(center_x) == 0:
            return None
        
        # 2. 多项式拟合
        coeffs = self.fit_polynomial(center_x, center_y)
        
        if coeffs is None:
            return None
        
        # 3. 生成采样点
        sampled_y = np.arange(start_x, end_x, step)
        sampled_x = np.polyval(coeffs, sampled_y)
        
        # 4. 计算一阶导和二阶导
        coeffs_d1 = np.polyder(coeffs, 1)
        coeffs_d2 = np.polyder(coeffs, 2)
        
        dy = np.polyval(coeffs_d1, sampled_y)
        ddy = np.polyval(coeffs_d2, sampled_y)
        
        # 5. 计算航向角和曲率
        yaw = np.arctan(dy)
        curvature = ddy / (1.0 + dy ** 2) ** 1.5
        
        return {
            "coeffs": coeffs,
            "center_x": center_x,
            "center_y": center_y,
            "sampled_x": sampled_x,
            "sampled_y": sampled_y,
            "yaw": yaw,
            "curvature": curvature
        }
    
    def fit_mask(self, bev_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        简化版：仅返回多项式系数
        
        Args:
            bev_mask: 鸟瞰图掩码
        
        Returns:
            多项式系数，失败返回None
        """
        center_x, center_y = self.extract_center_points(bev_mask)
        
        if len(center_x) == 0:
            return None
        
        return self.fit_polynomial(center_x, center_y)
    
    def get_coeffs_for_uart(self, bev_mask: np.ndarray) -> Optional[list]:
        """
        获取用于UART发送的多项式系数
        
        Args:
            bev_mask: 鸟瞰图掩码
        
        Returns:
            [a, b, c, d] 格式的系数列表，失败返回None
        """
        coeffs = self.fit_mask(bev_mask)
        
        if coeffs is None or len(coeffs) != 4:
            return None
        
        return coeffs.tolist()


# ============================================
# 单例模式
# ============================================
_fitter_instance = None

def get_lane_fitter(poly_order: int = None) -> LaneFitter:
    """获取车道线拟合器单例"""
    global _fitter_instance
    if _fitter_instance is None:
        _fitter_instance = LaneFitter(poly_order)
    return _fitter_instance


# ============================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("测试车道线拟合器...")
    
    try:
        fitter = LaneFitter(poly_order=3)
        
        # 创建测试掩码（模拟一条车道线）
        test_mask = np.zeros((720, 1280), dtype=np.uint8)
        
        # 绘制一条模拟曲线
        y = np.arange(0, 720)
        # x = 0.0001*y^3 - 0.1*y^2 + 10*y + 400
        x = 0.0001 * y**3 - 0.1 * y**2 + 10 * y + 400
        x = x.astype(np.int32)
        
        # 绘制车道线（有一定宽度）
        for i in range(len(y)):
            if 0 <= x[i] < 1280:
                test_mask[y[i], x[i]] = 255
                # 添加一些宽度
                if x[i] + 1 < 1280:
                    test_mask[y[i], x[i] + 1] = 255
                if x[i] - 1 >= 0:
                    test_mask[y[i], x[i] - 1] = 255
        
        # 测试提取中心点
        print("提取中心点...")
        cx, cy = fitter.extract_center_points(test_mask)
        print(f"提取到 {len(cx)} 个中心点")
        
        # 测试多项式拟合
        print("多项式拟合...")
        coeffs = fitter.fit_polynomial(cx, cy)
        print(f"拟合系数: {coeffs}")
        
        # 测试完整流程
        print("完整拟合和采样...")
        result = fitter.fit_and_sample(test_mask)
        if result:
            print(f"多项式系数: {result['coeffs']}")
            print(f"采样点数: {len(result['sampled_x'])}")
            print(f"第一个点: ({result['sampled_x'][0]:.2f}, {result['sampled_y'][0]:.2f})")
        
        # 测试UART系数
        print("获取UART系数...")
        uart_coeffs = fitter.get_coeffs_for_uart(test_mask)
        print(f"UART系数: {uart_coeffs}")
        
        print("✅ 测试通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()