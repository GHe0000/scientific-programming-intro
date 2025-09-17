import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    """逻辑斯蒂映射函数"""
    return r * x * (1 - x)

def plot_system_behavior(r, x0, n_iter=100):
    """
    分析并绘制给定 r 和 x0 下的相空间、轨道和蛛网图。
    """
    # --- 数据准备 ---
    # 1. 生成轨道数据
    orbit = np.zeros(n_iter + 1)
    orbit[0] = x0
    for i in range(n_iter):
        orbit[i+1] = logistic_map(r, orbit[i])

    # 2. 准备相空间背景图数据
    x_space = np.linspace(0, 1, 400)
    fx = logistic_map(r, x_space)

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Logistic Map Behavior for r = {r}, x₀ = {x0}', fontsize=16)

    # --- 图 1: 时间序列图 (轨道) ---
    ax1 = axes[0]
    ax1.plot(range(n_iter + 1), orbit, 'b-o', markersize=3, alpha=0.7)
    ax1.set_title('Time Series Plot (Orbit)', fontsize=14)
    ax1.set_xlabel('Iteration (n)', fontsize=12)
    ax1.set_ylabel('State (xₙ)', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 图 2: 相空间与蛛网图 ---
    ax2 = axes[1]
    # 绘制背景：抛物线和对角线
    ax2.plot(x_space, fx, 'r-', lw=2, label='f(x) = rx(1-x)')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='y = x')

    # 绘制蛛网图
    # 忽略前几个点，让轨道先进入吸引子
    start_plot_index = max(0, n_iter - 100) 
    for i in range(start_plot_index, n_iter):
        # 垂直线: (x_i, x_i) -> (x_i, x_{i+1})
        ax2.plot([orbit[i], orbit[i]], [orbit[i], orbit[i+1]], 'g-', lw=0.8)
        # 水平线: (x_i, x_{i+1}) -> (x_{i+1}, x_{i+1})
        ax2.plot([orbit[i], orbit[i+1]], [orbit[i+1], orbit[i+1]], 'g-', lw=0.8)

    ax2.set_title('Phase Space & Cobweb Plot', fontsize=14)
    ax2.set_xlabel('Current State (xₙ)', fontsize=12)
    ax2.set_ylabel('Next State (xₙ₊₁)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    # 初始状态点
    initial_x0 = 0.2

    # 案例 1: r = 2.8 (收敛到稳定的不动点)
    plot_system_behavior(r=2.8, x0=initial_x0)

    # 案例 2: r = 3.2 (收敛到 2-周期循环)
    plot_system_behavior(r=3.2, x0=initial_x0)

    # 案例 3: r = 3.5 (收敛到 4-周期循环)
    plot_system_behavior(r=3.5, x0=initial_x0)

    # 案例 4: r = 3.8 (混沌状态)
    plot_system_behavior(r=3.8, x0=initial_x0)
    
    # 案例 5: 混沌状态下初始值的敏感性
    # plot_system_behavior(r=3.8, x0=initial_x0 + 1e-9) 
    # (可以取消注释，对比案例4，你会发现轨道完全不同)
