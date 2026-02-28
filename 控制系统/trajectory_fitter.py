import numpy as np
from typing import Dict, Optional


class LaneTrajectoryFitter:
    def __init__(self, poly_order: int = 3):
        """
        初始化轨迹拟合器

        参数:
            poly_order: 多项式阶数。默认3阶，足以应对大多数直道和常规弯道。
                        如果算力吃紧且仅有平缓弯道，可降为2阶。
        """
        self.poly_order = poly_order

    def fit_and_sample(self,
                       pts_x: np.ndarray,
                       pts_y: np.ndarray,
                       start_x: float,
                       end_x: float,
                       step: float) -> Optional[Dict[str, np.ndarray]]:
        """
        对物理坐标系下的离散点进行多项式拟合，并按固定纵向步长采样。
        全部采用 NumPy 向量化运算以满足 >= 30 FPS 的边缘端实时性要求。

        参数:
            pts_x: 观测到的车道线点集 X 坐标 (前方距离)
            pts_y: 观测到的车道线点集 Y 坐标 (横向偏移)
            start_x: 采样起始距离 (例如车头前方 0.0m)
            end_x: 采样终止距离 (例如车头前方 15.0m)
            step: 采样步长 (例如 0.1m)

        返回:
            包含控制模块所需状态量的字典: 'x', 'y', 'yaw', 'k', 以及拟合系数 'coeffs'。
            若点数不足以拟合，返回 None。
        """
        # 容错处理：确保输入点数足够进行多项式拟合
        if len(pts_x) < self.poly_order + 1:
            # 未完成：实际接入 main_controller 时可替换为 logging.warning
            print("[Warning] Not enough points to fit the polynomial.")
            return None

        # 1. 多项式拟合 (返回系数按高次到低次排列: p0, p1, ..., pn)
        # 内部使用 SVD 分解，鲁棒性较好
        coeffs = np.polyfit(pts_x, pts_y, self.poly_order)

        # 2. 解析求导获取一阶导和二阶导函数的系数
        coeffs_d1 = np.polyder(coeffs, 1)
        coeffs_d2 = np.polyder(coeffs, 2)

        # 3. 生成按固定步长分布的 X 序列
        sampled_x = np.arange(start_x, end_x, step)

        # 4. 向量化计算所有的 Y, Y', Y''
        sampled_y = np.polyval(coeffs, sampled_x)
        dy = np.polyval(coeffs_d1, sampled_x)
        ddy = np.polyval(coeffs_d2, sampled_x)

        # 5. 计算航向角 (yaw) 和曲率 (k)
        yaw = np.arctan(dy)
        k = ddy / (1.0 + dy ** 2) ** 1.5

        return {
            "x": sampled_x,
            "y": sampled_y,
            "yaw": yaw,
            "k": k,
            "coeffs": coeffs  # 保留系数，方便后续 Matplotlib 可视化模块调用
        }


# ==========================================
# 快速模块测试 (Mock 数据)
# ==========================================
if __name__ == "__main__":
    # 模拟 BEV 视角下获取的带噪声的车道线像素/物理坐标
    mock_pts_x = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    # 模拟一个轻微左弯
    mock_pts_y = np.array([0.1, 0.2, 0.4, 0.8, 1.4, 2.2])

    fitter = LaneTrajectoryFitter(poly_order=2)

    # 假设我们要在前方 0m 到 12m 范围内，每隔 0.5m 采样一个点给 MPC
    trajectory = fitter.fit_and_sample(
        pts_x=mock_pts_x,
        pts_y=mock_pts_y,
        start_x=0.0,
        end_x=12.0,
        step=0.5
    )

    if trajectory:
        print(f"Generated {len(trajectory['x'])} reference points.")
        print(
            f"Sample X[0]: {trajectory['x'][0]:.2f}, Y[0]: {trajectory['y'][0]:.2f}, Yaw[0]: {trajectory['yaw'][0]:.4f}, K[0]: {trajectory['k'][0]:.4f}")