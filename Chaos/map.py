import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def logistic_map(r, x):
    return r * x * (1 - x)

def sin_map(r, x):
    return r * np.sin(np.pi * x)

def feigenbaum_diagram(map_func, r_range=(1.0, 4.0), x0=0.2, n_r=2000, n_iter=1000, n_plot=300):
    r = np.linspace(r_range[0], r_range[1], n_r)
    x = x0 * np.ones(n_r)
    
    for _ in range(n_iter - n_plot):
        x = map_func(r, x)
        
    result = np.zeros((n_plot, n_r))
    for i in range(n_plot):
        x = map_func(r, x)
        result[i] = x
        
    plt.figure(figsize=(12, 8))
    plt.plot(r, result.T, ',k', alpha=0.1)
    
    plt.title(f"Feigenbaum Diagram for '{map_func.__name__}' (Vectorized)", fontsize=16)
    plt.xlabel("Parameter (r)", fontsize=12)
    plt.ylabel("State (x)", fontsize=12)
    plt.xlim(r_range)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def find_bifurcation_points(map_func, r_start, x_max_func, max_period=128, r_tol=1e-9, n_iter_stable=2000):
    r_n = []
    attractors = {}
    r = r_start
    period = 2

    # [迭代] 外层循环：按顺序寻找 2-周期, 4-周期, 8-周期的分叉点
    while period <= max_period:
        r_low, r_high = r, r + 0.5  # 初始化搜索范围
        
        # [迭代] 内层循环：二分搜索，每一步依赖于上一步的结果
        while r_high - r_low > r_tol:
            r_mid = (r_high + r_low) / 2.0
            x = x_max_func(r_mid)
            # [迭代] 热身，让系统收敛到吸引子
            for _ in range(n_iter_stable):
                x = map_func(r_mid, x)
            
            # [迭代] 收集吸引子上的点
            points = deque(maxlen=period * 2)
            for _ in range(period * 2):
                x = map_func(r_mid, x)
                points.append(round(x, 10))
            current_period = len(set(points))
            if current_period >= period:
                r_high = r_mid  # 周期已倍增，向左搜索
            else:
                r_low = r_mid   # 周期未倍增，向右搜索
        r = r_high  # 收敛的分叉点
        r_n.append(r)
        
        # --- 记录分叉后的吸引子 (同样是针对单个r值的迭代) ---
        r_attractor = r + 1e-7
        x = x_max_func(r_attractor)
        for _ in range(n_iter_stable):
            x = map_func(r_attractor, x)
        
        attractor_points = [map_func(r_attractor, x) for _ in range(period)]
        attractors[period] = sorted(list(set(round(p, 10) for p in attractor_points)))
        # --- 结束记录 ---

        print(f"Found bifurcation from period {period//2} to {period} at r = {r:.10f}")
        period *= 2
        
    return r_n, attractors

# --- 4. 计算费根鲍姆常数的函数 (只是简单的数学运算，非性能瓶颈) ---
def calculate_feigenbaum_constants(r_n, attractors, x_max_func):
    """根据分叉点和吸引子计算费根鲍姆常数。此函数本身不涉及迭代，只是后处理。"""
    
    # [向量化] 使用 NumPy 进行数组运算，更简洁高效
    r_n_arr = np.array(r_n)
    
    # 计算 Delta
    # (r_{n} - r_{n-1}) / (r_{n+1} - r_{n})
    deltas = (r_n_arr[1:-1] - r_n_arr[:-2]) / (r_n_arr[2:] - r_n_arr[1:-1])
    
    # 计算 Alpha
    alphas = []
    periods = sorted(attractors.keys())
    if len(periods) >= 2:
        x_c = x_max_func(3.0) # r值不重要
        for i in range(len(periods) - 1):
            d_n = np.min(np.abs(np.array(attractors[periods[i]]) - x_c))
            d_n1 = np.min(np.abs(np.array(attractors[periods[i+1]]) - x_c))
            if d_n1 > 1e-12: # 避免除以零
                alphas.append(-d_n / d_n1) # 加上负号符合定义
    
    return deltas, alphas


# --- 5. 主程序入口 ---
if __name__ == '__main__':
    print("--- Analyzing the Logistic Map ---")
    
    # 1. 向量化绘图
    feigenbaum_diagram(logistic_map, r_range=(0.0, 4.0))

    # 2. 迭代搜索分叉点
    logistic_x_max = lambda r: 0.5
    r_n_log, attractors_log = find_bifurcation_points(
        logistic_map, r_start=3.0, x_max_func=logistic_x_max, max_period=256
    )
    
    # 3. 后处理计算常数
    delta_log, alpha_log = calculate_feigenbaum_constants(r_n_log, attractors_log, logistic_x_max)
    
    print("\nLogistic Map Feigenbaum Constant Estimates:")
    print("Delta (δ) -> 4.6692...")
    for i, d in enumerate(delta_log):
        print(f"  δ_{i+2} = {d:.6f}") # n从2开始
        
    print("\nAlpha (α) -> -2.5029...")
    for i, a in enumerate(alpha_log):
        print(f"  α_{i+1} = {a:.6f}")
