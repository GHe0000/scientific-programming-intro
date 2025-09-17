import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    """逻辑斯蒂映射函数"""
    return r * x * (1 - x)

def feigenbaum_with_boundaries(r_min=2.5, r_max=4.0, n_r=4000, n_iter=2000, n_plot=300):
    """
    生成并绘制费根鲍姆图，并叠加其解析边界。
    """
    # --- 1. 绘制费根鲍姆图 (与之前的代码相同) ---
    r = np.linspace(r_min, r_max, n_r)
    x = 1e-5 * np.ones(n_r)
    
    # 热身
    for _ in range(n_iter - n_plot):
        x = logistic_map(r, x)
        
    # 收集数据
    result = np.zeros((n_plot, n_r))
    for i in range(n_plot):
        x = logistic_map(r, x)
        result[i] = x
        
    plt.figure(figsize=(14, 9))
    plt.plot(r, result.T, ',k', alpha=0.1)
    
    # --- 2. 计算并绘制解析边界 ---
    
    # 定义一个 r 的范围用于绘制平滑曲线，特别是在混沌区
    r_chaos = np.linspace(3.56995, 4.0, 1000)
    
    # 极大值点 (critical point)
    xc = 0.5 
    
    # 迭代 1: 上边界
    y1 = logistic_map(r_chaos, xc)
    
    # 迭代 2: 下边界
    y2 = logistic_map(r_chaos, y1)
    
    # 迭代 3: 内部上边界
    y3 = logistic_map(r_chaos, y2)

    # 迭代 4: 内部下边界
    y4 = logistic_map(r_chaos, y3)

    plt.plot(r_chaos, y1, 'r-', lw=2, label='$y = f(x_c)$ (Upper Bound)')
    plt.plot(r_chaos, y2, 'b-', lw=2, label='$y = f^2(x_c)$ (Lower Bound)')
    plt.plot(r_chaos, y3, 'g-', lw=1.5, alpha=0.8, label='$y = f^3(x_c)$ (Internal Boundary)')
    plt.plot(r_chaos, y4, 'c-', lw=1.5, alpha=0.8, label='$y = f^4(x_c)$ (Internal Boundary)')

    plt.title("Feigenbaum Diagram with Analytical Boundaries", fontsize=16)
    plt.xlabel("Growth Rate (r)", fontsize=12)
    plt.ylabel("Population (x)", fontsize=12)
    plt.xlim(r_min, r_max)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    feigenbaum_with_boundaries()
