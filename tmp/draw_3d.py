import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ==============================================================================
# 核心计算函数 (使用 Numba JIT 编译)
# ==============================================================================

@njit(cache=True)
def lorenz_deriv(state, t, sigma, beta, r):
    """
    计算洛伦兹系统的导数。
    state: 状态向量 (x, y, z)
    t: 时间 (洛伦兹系统不依赖于时间，但这是 ODE 求解器的标准形式)
    sigma, beta, r: 系统参数
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (r - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

@njit(cache=True)
def solve_lorenz_and_get_section(r, sigma, beta, initial_state, dt, num_steps, transient_steps):
    """
    为单个 'r' 值求解洛伦兹系统，并找到在 z = r - 1 平面上的庞加莱截面。
    我们只记录当轨迹从下往上穿过该平面 (dz/dt > 0) 时的点。
    """
    state = initial_state.copy()
    
    # 瞬态演化：运行一段时间让轨迹稳定到吸引子上
    for _ in range(transient_steps):
        # 四阶龙格-库塔 (RK4) 积分步
        k1 = dt * lorenz_deriv(state, 0, sigma, beta, r)
        k2 = dt * lorenz_deriv(state + 0.5 * k1, 0, sigma, beta, r)
        k3 = dt * lorenz_deriv(state + 0.5 * k2, 0, sigma, beta, r)
        k4 = dt * lorenz_deriv(state + k3, 0, sigma, beta, r)
        state += (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # 预分配一个足够大的数组来存储截面上的点
    max_points = 20000 
    section_points = np.empty((max_points, 2), dtype=np.float64)
    point_count = 0
    
    # 定义截面平面
    z_plane = r - 1.0

    # 主循环：继续积分并寻找交点
    for i in range(num_steps):
        state_prev = state.copy()
        z_prev = state_prev[2]

        # RK4 积分步
        k1 = dt * lorenz_deriv(state, 0, sigma, beta, r)
        k2 = dt * lorenz_deriv(state + 0.5 * k1, 0, sigma, beta, r)
        k3 = dt * lorenz_deriv(state + 0.5 * k2, 0, sigma, beta, r)
        k4 = dt * lorenz_deriv(state + k3, 0, sigma, beta, r)
        state += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        z_curr = state[2]

        # 检查是否从下往上穿过平面 (z_prev < z_plane <= z_curr)
        if z_prev < z_plane and z_curr >= z_plane:
            # 使用线性插值来精确计算交点的 (x, y) 坐标
            s = (z_plane - z_prev) / (z_curr - z_prev)
            x_intersect = state_prev[0] + s * (state[0] - state_prev[0])
            y_intersect = state_prev[1] + s * (state[1] - state_prev[1])

            # 存储结果
            if point_count < max_points:
                section_points[point_count, 0] = x_intersect
                section_points[point_count, 1] = y_intersect
                point_count += 1

    # 只返回实际找到的点
    return section_points[:point_count]


@njit(parallel=True, cache=True)
def compute_poincare_parallel(r_values, sigma, beta, initial_state, dt, num_steps, transient_steps):
    """
    并行计算一系列 'r' 值的庞加莱截面。
    返回一个列表，其中每个元素是对应 'r' 值的 (x, y) 坐标数组。
    """
    n_r = len(r_values)
    results = [np.empty((0, 2), dtype=np.float64) for _ in range(n_r)]

    # 使用 prange 进行并行循环
    for i in prange(n_r):
        r = r_values[i]
        perturbed_initial_state = initial_state + 1e-5 * i
        
        # ==================== FIX HERE ====================
        # 将函数名从 solve_lorenz_for_section 改为 solve_lorenz_and_get_section
        points = solve_lorenz_and_get_section(r, sigma, beta, perturbed_initial_state, dt, num_steps, transient_steps)
        # ================================================

        results[i] = points
        
    return results


# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 定义参数 ---
    sigma = 10.0
    beta = 8.0 / 3.0
    
    r_min = 10.0
    r_max = 25.0 
    num_r_points = 500
    r_values = np.linspace(r_min, r_max, num_r_points)

    dt = 0.01              
    transient_steps = 2000 
    num_steps = 40000      

    initial_state = np.array([1.0, 1.0, 1.0])

    # --- 2. 执行计算 ---
    print("开始计算... (首次运行需要编译 Numba 函数，可能较慢)")
    
    start_time = time.time()
    results_list = compute_poincare_parallel(r_values, sigma, beta, initial_state, dt, num_steps, transient_steps)
    end_time = time.time()
    
    print(f"计算完成！总耗时: {end_time - start_time:.2f} 秒。")

    # --- 3. 数据后处理 ---
    print("正在处理数据用于绘图...")
    total_points = sum(len(arr) for arr in results_list)
    plot_r = np.zeros(total_points)
    plot_x = np.zeros(total_points)
    plot_y = np.zeros(total_points)
    
    current_idx = 0
    for i, r_val in enumerate(r_values):
        points = results_list[i]
        num_found_points = len(points)
        if num_found_points > 0:
            start = current_idx
            end = current_idx + num_found_points
            plot_r[start:end] = r_val
            plot_x[start:end] = points[:, 0]
            plot_y[start:end] = points[:, 1]
            current_idx = end

    print(f"共收集到 {total_points} 个截面点。")

    # --- 4. 三维可视化 ---
    if total_points > 0:
        print("正在生成三维图像...")
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(plot_r, plot_x, plot_y, s=0.1, c=plot_y, cmap='plasma', marker='.')

        ax.set_xlabel('Parameter $r$', fontsize=12)
        ax.set_ylabel('$x$ coordinate', fontsize=12)
        ax.set_zlabel('$y$ coordinate', fontsize=12)
        ax.set_title('Poincaré Section ($z=r-1$) of the Lorenz System', fontsize=16)
        
        ax.view_init(elev=25., azim=-75)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.show()
    else:
        print("没有收集到任何截面点，请检查参数设置。")
