import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math
from scipy.linalg import solve_discrete_are


# --- 1. 参数设置 (包含动力学参数) ---
class Params:
    def __init__(self):
        self.dt = 0.1
        self.N = 10
        self.L = 2.5
        self.v_ref = 10.0

        # 动力学参数
        self.m = 1500.0
        self.Iz = 2250.0
        self.Cf = 160000.0
        self.Cr = 180000.0
        self.lf = 1.1
        self.lr = 1.4

        # 车辆限制
        self.max_steer = np.deg2rad(30)
        self.max_dsteer = np.deg2rad(15)

        # 权重矩阵
        self.Q = np.diag([20.0, 1.0, 20.0, 1.0])
        self.R = np.diag([5.0])

    # --- 2. 轨迹生成 ---


class ReferencePath:
    def __init__(self):
        self.x = np.linspace(0, 100, 1000)
        self.y = 5.0 * np.sin(self.x / 10.0)
        self.yaw = np.gradient(self.y, self.x)
        self.k = np.gradient(self.yaw, self.x)

    def calc_track_error(self, x, y, yaw):
        d = np.hypot(self.x - x, self.y - y)
        idx = np.argmin(d)

        dx = x - self.x[idx]
        dy = y - self.y[idx]
        vec_path = np.array([np.cos(self.yaw[idx]), np.sin(self.yaw[idx])])
        vec_err = np.array([dx, dy])
        cross = vec_path[0] * vec_err[1] - vec_path[1] * vec_err[0]
        e = math.copysign(np.hypot(dx, dy), cross)

        th_e = yaw - self.yaw[idx]
        while th_e > np.pi: th_e -= 2 * np.pi
        while th_e < -np.pi: th_e += 2 * np.pi

        return e, th_e, idx, self.k[idx]


# --- 3. 动力学模型 (修复后的版本) ---
def get_linear_model(v, dt, p):
    v = max(v, 0.1)  # 避免除零
    Cf, Cr, m, Iz = p.Cf, p.Cr, p.m, p.Iz
    lf, lr = p.lf, p.lr

    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 1] = -(Cf + Cr) / (m * v)
    A[1, 2] = (Cf + Cr) / m
    A[1, 3] = (lr * Cr - lf * Cf) / (m * v)
    A[2, 3] = 1.0
    A[3, 1] = (lr * Cr - lf * Cf) / (Iz * v)
    A[3, 2] = (lf * Cf - lr * Cr) / Iz
    A[3, 3] = -(lf ** 2 * Cf + lr ** 2 * Cr) / (Iz * v)

    B = np.zeros((4, 1))
    B[1, 0] = Cf / m
    B[3, 0] = lf * Cf / Iz

    Ad = np.eye(4) + A * dt
    Bd = B * dt
    return Ad, Bd


# --- 4. MPC + LQR 控制器 (增强鲁棒性版) ---
class MPC_LQR_Controller:
    def __init__(self, params):
        self.p = params

    def solve_lqr_gain(self, Ad, Bd):
        P = solve_discrete_are(Ad, Bd, self.p.Q, self.p.R)
        K = np.linalg.inv(self.p.R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        return K, P

    def solve(self, x0, Ad, Bd):
        K_lqr, P_lqr = self.solve_lqr_gain(Ad, Bd)

        nx, nu = 4, 1
        x = cp.Variable((nx, self.p.N + 1))
        u = cp.Variable((nu, self.p.N))

        cost = 0.0
        constraints = [x[:, 0] == x0]

        for t in range(self.p.N):
            cost += cp.quad_form(x[:, t], self.p.Q) + cp.quad_form(u[:, t], self.p.R)
            constraints += [x[:, t + 1] == Ad @ x[:, t] + Bd @ u[:, t]]
            constraints += [cp.abs(u[:, t]) <= self.p.max_steer]
            if t > 0:
                constraints += [cp.abs(u[:, t] - u[:, t - 1]) <= self.p.max_dsteer * self.p.dt]

        cost += cp.quad_form(x[:, self.p.N], P_lqr)

        prob = cp.Problem(cp.Minimize(cost), constraints)

        # === 修复 1: 放宽求解精度，避免 inaccurate 警告 ===
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-3, eps_rel=1e-3, verbose=False)
        except Exception as e:
            print(f"Solver Crash: {e}")
            prob.status = 'solver_error'

        # === 修复 2: 严格的有效性检查 ===
        # 如果状态不是最优，或者计算出的 u 是 None/NaN，则回退到 LQR
        use_mpc = False
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if u.value is not None and not np.isnan(u.value[0, 0]):
                use_mpc = True

        if use_mpc:
            return u.value[0, 0], x.value
        else:
            # 备用方案：直接用 LQR 增益算控制量
            # print("Warning: MPC failed/unstable, switching to LQR fallback")
            u_fallback = -K_lqr @ x0
            return u_fallback[0], None


# --- 5. 仿真主循环 ---
def run_simulation():
    p = Params()
    path = ReferencePath()
    controller = MPC_LQR_Controller(p)

    # 初始状态
    x, y, yaw = 0.0, 1.0, 0.0
    v = p.v_ref

    # 记录历史数据
    history_x = [x]
    history_y = [y]
    history_steer = []

    print("Simulating...", end="")
    for i in range(300):  # 增加步数，跑远一点
        # 1. 计算误差
        e, th_e, idx, curvature = path.calc_track_error(x, y, yaw)

        # 2. 准备状态和模型
        state_vector = np.array([e, 0.0, th_e, 0.0])
        Ad, Bd = get_linear_model(v, p.dt, p)

        # 3. 计算控制量
        steer_mpc, _ = controller.solve(state_vector, Ad, Bd)

        # 4. 前馈补偿
        steer_ff = math.atan(p.L * curvature)
        final_steer = steer_mpc + steer_ff

        # 5. 安全检查
        if np.isnan(final_steer):
            print(f"\nSimulation stopped: NaN detected at step {i}")
            break

        # 6. 更新车辆动力学
        x += v * np.cos(yaw) * p.dt
        y += v * np.sin(yaw) * p.dt
        yaw += v / p.L * np.tan(final_steer) * p.dt

        history_x.append(x)
        history_y.append(y)
        history_steer.append(final_steer)

        if i % 50 == 0: print(".", end="", flush=True)  # 打印进度点

    print(" Done!")

    # --- 仿真结束后一次性绘图 ---
    plt.figure(figsize=(12, 6))

    # 1. 画轨迹
    plt.plot(path.x, path.y, "k--", label="Reference Path")
    plt.plot(history_x, history_y, "r-", linewidth=2, label="MPC+LQR Trajectory")
    plt.scatter([history_x[0]], [history_y[0]], color='b', label="Start")
    plt.scatter([history_x[-1]], [history_y[-1]], color='g', marker='*', s=200, label="End")

    plt.title(f"Simulation Result: Steps={len(history_x)} | Final Error={e:.3f}m")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_simulation()