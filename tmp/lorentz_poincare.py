import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit(nopython=True)
def lorenz(t, xyz, sigma, rho, beta):
    """
    洛伦兹系统的微分方程。
    """
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# --- 2. 实现固定步长的 RK4 求解器 ---

@jit(nopython=True)
def rk4_solver_poincare(
    func, t_span, y0, h, z_section, args
):
    t_start, t_end = t_span
    t = t_start
    y = y0.copy()
    
    num_steps = int((t_end - t_start) / h)
    poincare_points = []
    
    # 预热/瞬态过程，让轨迹先进入吸引子，避免初始状态的影响
    transient_steps = int(20.0 / h) # 假设预热20个时间单位
    for _ in range(transient_steps):
        k1 = func(t, y, *args)
        k2 = func(t + 0.5*h, y + 0.5*h*k1, *args)
        k3 = func(t + 0.5*h, y + 0.5*h*k2, *args)
        k4 = func(t + h, y + h*k3, *args)
        y += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h

    print("预热完成，开始采集庞加莱截面点...")

    # 主循环，计算并采集截面点
    for step in range(num_steps - transient_steps):
        y_prev = y.copy()
        
        # --- RK4 核心计算 ---
        k1 = func(t, y, *args)
        k2 = func(t + 0.5*h, y + 0.5*h*k1, *args)
        k3 = func(t + 0.5*h, y + 0.5*h*k2, *args)
        k4 = func(t + h, y + h*k3, *args)
        
        y_curr = y_prev + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # --- 检查是否穿越庞加莱截面 (从下往上) ---
        z_prev = y_prev[2]
        z_curr = y_curr[2]
        
        if z_prev < z_section and z_curr >= z_section:
            # 线性插值找到精确的交点
            # ratio = (z_target - z_start) / (z_end - z_start)
            ratio = (z_section - z_prev) / (z_curr - z_prev)
            px = y_prev[0] + ratio * (y_curr[0] - y_prev[0])
            py = y_prev[1] + ratio * (y_curr[1] - y_prev[1])
            poincare_points.append((px, py))

        # 更新状态
        y = y_curr
        t += h
        
    return poincare_points


# --- 3. 主程序：设置参数并运行 ---
if __name__ == "__main__":
    # 洛伦兹系统参数
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    # 初始条件和积分设置
    initial_state = np.array([0.0, 1.0, 1.05], dtype=np.float64)
    time_span = [0.0, 2000.0]
    
    # --- 关键参数：固定步长 ---
    # 步长 h 需要足够小以保证精度。h=0.001 是一个比较安全的选择。
    # 你可以尝试 h=0.01，可能会发现图形有些许差异或变得粗糙。
    step_size_h = 0.001
    
    # 定义截面
    section = rho - 1

    print(f"正在使用固定步长 h={step_size_h} 的 RK4 求解器计算...")
    start_time = time.time()
    
    # 调用求解器
    points = rk4_solver_poincare(
        lorenz, time_span, initial_state, step_size_h, section,
        args=(sigma, rho, beta)
    )
    
    end_time = time.time()
    print(f"计算完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"共采集到 {len(points)} 个截面点。")

    # --- 4. 绘图 ---
    if points:
        # 将点列表转换为 Numpy 数组方便绘图
        poincare_data = np.array(points)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(poincare_data[:, 0], poincare_data[:, 1], s=1, c='darkred', alpha=0.7)
        plt.title(f"Lorentz System Poincaré Section (Fixed-Step RK4, h={step_size_h}) at z = {section}", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("y", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.show()
    else:
        print("未能采集到任何庞加莱截面点，请检查参数或积分时间。")
