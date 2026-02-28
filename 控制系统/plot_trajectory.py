import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from trajectory_fitter import LaneTrajectoryFitter

def visualize_trajectory():
    # 1. 准备和刚才一样的 Mock 数据
    mock_pts_x = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    mock_pts_y = np.array([0.1, 0.2, 0.4, 0.8, 1.4, 2.2])

    fitter = LaneTrajectoryFitter(poly_order=2)
    trajectory = fitter.fit_and_sample(
        pts_x=mock_pts_x,
        pts_y=mock_pts_y,
        start_x=0.0,
        end_x=12.0,
        step=0.5
    )

    if not trajectory:
        print("拟合失败！")
        return

    x = trajectory['x']
    y = trajectory['y']
    yaw = trajectory['yaw']
    k = trajectory['k']

    # 2. 绘制三合一图表
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # X-Y 轨迹图
    axs[0].scatter(mock_pts_x, mock_pts_y, color='red', label='Observed Points (Mask/IPM)')
    axs[0].plot(x, y, color='blue', label='Fitted Trajectory')
    axs[0].set_title('Lane Trajectory (X-Y)')
    axs[0].set_xlabel('X (Forward Distance [m])')
    axs[0].set_ylabel('Y (Lateral Offset [m])')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal') # 保持XY比例一致，真实反映弯道形状

    # Yaw 航向角图
    axs[1].plot(x, np.degrees(yaw), color='green')
    axs[1].set_title('Heading Angle (Yaw)')
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Yaw [Degrees]')
    axs[1].grid(True)

    # Curvature 曲率图
    axs[2].plot(x, k, color='purple')
    axs[2].set_title('Curvature (k)')
    axs[2].set_xlabel('X [m]')
    axs[2].set_ylabel('Curvature [1/m]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_trajectory()