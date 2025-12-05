import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit, prange

@njit(parallel=True)
def compute_period_map(xmin, xmax, ymin, ymax, width, height, max_iter, tol, max_period):
    """
    计算 Mandelbrot 周期图。
    Returns:
        result: 2D array
        0: 外部 (Escaped)
        1~max_period: 对应的周期
        -1: 内部但未检测到指定范围内的周期 (可能是更高周期或混沌)
    """
    result = np.zeros((height, width), dtype=np.int32)
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    
    for y in prange(height):
        c_im = ymin + y * dy
        for x in range(width):
            c_re = xmin + x * dx
            c = complex(c_re, c_im)
            
            z = 0j
            escaped = False
            
            # 1. 逃逸检测
            for _ in range(max_iter):
                if z.real*z.real + z.imag*z.imag > 4.0:
                    escaped = True
                    break
                z = z*z + c
            
            if escaped:
                result[y, x] = 0
            else:
                # 2. 周期检测
                z_snapshot = z
                found_p = -1
                for p in range(1, max_period + 1):
                    z = z*z + c
                    if abs(z - z_snapshot) < tol:
                        found_p = p
                        break
                result[y, x] = found_p
                
    return result

def plot_discrete_period_map(data, xmin, xmax, ymin, ymax, max_period):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # --- 颜色策略 ---
    # 0 (外部): 黑色
    # -1 (未知内部): 灰色
    # 1~N (周期): 使用 tab20 或其他高对比度色图
    
    # 1. 首先绘制背景（外部）和未知内部区域
    # 创建一个全黑底色
    ax.set_facecolor('black') 
    
    # 将 -1 (未知内部) 绘制为灰色
    # 使用 masked array 只显示 -1 的部分
    mask_unknown = np.ma.masked_where(data != -1, data)
    ax.imshow(mask_unknown, extent=[xmin, xmax, ymin, ymax], origin='lower',
              cmap=mcolors.ListedColormap(['dimgray']), interpolation='nearest')
    
    # 2. 绘制周期区域 (1 到 max_period)
    # 提取 tab20 颜色中的前 max_period 种颜色
    # 如果周期很多，可以循环使用颜色
    base_cmap = plt.get_cmap('tab20', max_period)
    color_list = [base_cmap(i) for i in range(max_period)]
    cmap = mcolors.ListedColormap(color_list)
    
    # 定义边界：[0.5, 1.5, 2.5, ... max_period + 0.5]
    # 这样整数 1 就会落在 0.5~1.5 之间，对应第一种颜色
    bounds = np.arange(0.5, max_period + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # 掩盖掉非周期区域 (0 和 -1)，只绘制 1~max_period
    mask_periods = np.ma.masked_less(data, 1)
    
    im = ax.imshow(mask_periods, extent=[xmin, xmax, ymin, ymax], origin='lower',
                   cmap=cmap, norm=norm, interpolation='nearest')
    
    # 3. 添加离散 Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(1, max_period + 1))
    cbar.set_label('Periodicity', rotation=270, labelpad=15)
    
    # 调整 Colorbar 的刻度位置，使其位于色块中心（BoundaryNorm 会自动处理，但显式设置 tick 更好）
    cbar.ax.tick_params(labelsize=10)

    # 装饰
    ax.set_title(f"Mandelbrot Set Regions by Period (1-{max_period})")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    
    plt.tight_layout()
    plt.savefig('mandelbrot_discrete_cbar.png')
    plt.show()

# --- 主程序执行 ---
# 参数设置
w, h = 1200, 1000
x_min, x_max = -2.0, 0.6
y_min, y_max = -1.2, 1.2
m_iter = 1000
tolerance = 1e-4
max_p = 12  # 检测前12个周期

print(f"正在计算周期 1 到 {max_p} ...")
period_data = compute_period_map(x_min, x_max, y_min, y_max, w, h, m_iter, tolerance, max_p)

print("正在绘图...")
plot_discrete_period_map(period_data, x_min, x_max, y_min, y_max, max_p)
