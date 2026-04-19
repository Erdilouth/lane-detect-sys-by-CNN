"""
工具函数模块
Utility Functions
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional


def draw_polyline(img: np.ndarray, 
                  coeffs: np.ndarray,
                  y_range: Tuple[int, int],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制多项式曲线
    
    Args:
        img: 输入图像
        coeffs: 多项式系数
        y_range: y坐标范围 (y_start, y_end)
        color: 颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    result = img.copy()
    
    y_start, y_end = y_range
    y_coords = np.arange(y_start, y_end)
    x_coords = np.polyval(coeffs, y_coords)
    
    # 转换为整数坐标
    pts = np.vstack((x_coords, y_coords)).astype(np.int32).T
    
    # 过滤有效点
    height, width = img.shape[:2]
    valid_pts = []
    for pt in pts:
        x, y = pt
        if 0 <= x < width and 0 <= y < height:
            valid_pts.append(pt)
    
    if len(valid_pts) > 1:
        cv2.polylines(
            result,
            [np.array(valid_pts)],
            isClosed=False,
            color=color,
            thickness=thickness
        )
    
    return result


def compute_curve_curvature(coeffs: np.ndarray, y: float) -> float:
    """
    计算多项式曲线在某点的曲率
    
    Args:
        coeffs: 多项式系数
        y: y坐标
    
    Returns:
        曲率值
    """
    # 一阶导
    coeffs_d1 = np.polyder(coeffs, 1)
    dy = np.polyval(coeffs_d1, y)
    
    # 二阶导
    coeffs_d2 = np.polyder(coeffs, 2)
    ddy = np.polyval(coeffs_d2, y)
    
    # 曲率公式
    curvature = ddy / (1.0 + dy ** 2) ** 1.5
    
    return curvature


def compute_curve_yaw(coeffs: np.ndarray, y: float) -> float:
    """
    计算多项式曲线在某点的航向角
    
    Args:
        coeffs: 多项式系数
        y: y坐标
    
    Returns:
        航向角（弧度）
    """
    # 一阶导
    coeffs_d1 = np.polyder(coeffs, 1)
    dy = np.polyval(coeffs_d1, y)
    
    # 航向角
    yaw = np.arctan(dy)
    
    return yaw


def create_colormap_mask(mask: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         alpha: float = 0.5) -> np.ndarray:
    """
    创建彩色掩码
    
    Args:
        mask: 二值掩码 (0或255)
        color: 颜色 (B, G, R)
        alpha: 透明度
    
    Returns:
        彩色掩码
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 设置颜色
    color_mask[mask > 0] = color
    
    return color_mask


def overlay_mask(img: np.ndarray,
                mask: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 0),
                alpha: float = 0.5) -> np.ndarray:
    """
    在图像上叠加掩码
    
    Args:
        img: 输入图像
        mask: 二值掩码
        color: 颜色 (B, G, R)
        alpha: 透明度
    
    Returns:
        叠加后的图像
    """
    color_mask = create_colormap_mask(mask, color)
    result = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
    
    return result


def filter_lanes_by_curvature(coeffs_list: list,
                             max_curvature: float = 0.1) -> list:
    """
    根据曲率过滤车道线
    
    Args:
        coeffs_list: 多项式系数列表
        max_curvature: 最大曲率阈值
    
    Returns:
        过滤后的系数列表
    """
    filtered = []
    
    for coeffs in coeffs_list:
        # 计算曲率（在中间位置）
        y = 360  # 假设图像高度为720
        curvature = abs(compute_curve_curvature(coeffs, y))
        
        if curvature <= max_curvature:
            filtered.append(coeffs)
    
    return filtered


def print_system_info():
    """打印系统信息"""
    import sys
    import platform
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("系统信息:")
    logger.info(f"  Python版本: {sys.version}")
    logger.info(f"  操作系统: {platform.system()} {platform.release()}")
    logger.info(f"  处理器: {platform.processor()}")
    
    # OpenCV版本
    logger.info(f"  OpenCV版本: {cv2.__version__}")
    
    # NumPy版本
    logger.info(f"  NumPy版本: {np.__version__}")
    
    # CUDA信息
    try:
        import tensorrt as trt
        logger.info(f"  TensorRT版本: {trt.__version__}")
    except:
        logger.warning("  TensorRT未安装")
    
    try:
        from cuda import cuda
        logger.info("  CUDA: 可用")
    except:
        logger.warning("  CUDA: 不可用")
    
    logger.info("=" * 50)


def benchmark_inference(engine, test_img: np.ndarray, 
                       iterations: int = 100) -> dict:
    """
    推理性能基准测试
    
    Args:
        engine: 推理引擎
        test_img: 测试图像
        iterations: 迭代次数
    
    Returns:
        性能统计信息
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始推理基准测试 ({iterations}次)...")
    
    times = []
    
    # 预热
    for _ in range(10):
        _ = engine.infer(test_img)
    
    # 正式测试
    for i in range(iterations):
        start = cv2.getTickCount()
        _ = engine.infer(test_img)
        end = cv2.getTickCount()
        
        time_ms = (end - start) / cv2.getTickFrequency() * 1000
        times.append(time_ms)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  进度: {i + 1}/{iterations}")
    
    times = np.array(times)
    
    stats = {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "fps": 1000.0 / np.mean(times)
    }
    
    logger.info("基准测试结果:")
    logger.info(f"  平均耗时: {stats['mean']:.2f} ms")
    logger.info(f"  标准差: {stats['std']:.2f} ms")
    logger.info(f"  最小耗时: {stats['min']:.2f} ms")
    logger.info(f"  最大耗时: {stats['max']:.2f} ms")
    logger.info(f"  中位数: {stats['median']:.2f} ms")
    logger.info(f"  平均FPS: {stats['fps']:.2f}")
    
    return stats



# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("测试工具函数...")
    
    # 测试多项式绘制
    print("测试多项式绘制...")
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    coeffs = np.array([0.0001, -0.1, 10, 400])
    result = draw_polyline(test_img, coeffs, (0, 720), (0, 255, 0), 5)
    print("✅ 多项式绘制测试通过")
    
    # 测试曲率计算
    print("测试曲率计算...")
    curvature = compute_curve_curvature(coeffs, 360)
    print(f"  曲率: {curvature:.6f}")
    
    # 测试航向角计算
    yaw = compute_curve_yaw(coeffs, 360)
    print(f"  航向角: {yaw:.6f} rad = {np.degrees(yaw):.2f} deg")
    
    # 测试掩码叠加
    print("测试掩码叠加...")
    test_mask = np.zeros((720, 1280), dtype=np.uint8)
    test_mask[300:420, 500:700] = 255
    result = overlay_mask(test_img, test_mask, (255, 0, 0), 0.5)
    print("✅ 掩码叠加测试通过")
    
    # print系统信息
    print("\n打印系统信息...")
    print_system_info()
    
    print("✅ 所有测试通过")