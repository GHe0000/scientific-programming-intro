import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_escape_time(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    计算 Mandelbrot 集合的逃逸时间。
    返回: 2D 数组 (int32)，值为逃逸所需的迭代次数。
    如果是 max_iter，则代表未逃逸（在集合内）。
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
            iter_count = 0
            
            for i in range(max_iter):
                # 优化: 检查是否逃逸 (R^2 > 4)
                if z.real*z.real + z.imag*z.imag > 4.0:
                    escaped = True
                    iter_count = i
                    break
                z = z*z + c
            
            if escaped:
                result[y, x] = iter_count
            else:
                # 集合内部，标记为 max_iter
                result[y, x] = max_iter

    return result

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_mandelbrot_escape(data, xmin, xmax, ymin, ymax, max_iter):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # --- 关键设置: 内部为白色 ---
    
    # 1. 创建掩码数组 (Masked Array)
    # 将所有值为 max_iter 的点标记为 "无效" (masked)
    # 这样它们就不会参与 colormap 的映射，而是显示底色或 set_bad 的颜色
    masked_data = np.ma.masked_equal(data, max_iter)
    
    # 2. 选择色图 (Colormap)
    # 'magma', 'inferno', 'twilight', 'hsv' 都是不错的选择
    # copy() 很重要，防止修改全局 colormap
    cmap = plt.cm.magma.copy() 
    
    # 3. 设置 "坏值" (Masked values) 的颜色为白色
    cmap.set_bad(color='white')
    
    # 4. 归一化 (Normalization)
    # 使用 PowerNorm 或 LogNorm 可以让边缘的渐变层次更丰富
    # gamma < 1 会提亮低值区域，让外部的黑色区域减少，细节更多
    norm = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=max_iter)
    
    # 5. 绘图
    im = ax.imshow(masked_data, extent=[xmin, xmax, ymin, ymax], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='bilinear')
    
    # 添加 Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Iterations to Escape (Speed)', rotation=270, labelpad=15)
    
    # 装饰
    ax.set_title(f"Mandelbrot Set Escape Time\n(White = Inside Set)")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    
    plt.tight_layout()
    plt.savefig('mandelbrot_escape_white.png')
    plt.show()

# ==========================================
# 主程序执行
# ==========================================

# 参数
width, height = 1600, 1400
xmin, xmax = -2.0, 0.6
ymin, ymax = -1.2, 1.2
max_iter = 500  # 迭代次数越多，边缘越精细，计算越慢

print("正在使用 Numba 计算逃逸时间...")
escape_data = compute_escape_time(xmin, xmax, ymin, ymax, width, height, max_iter)

print("正在渲染图像...")
plot_mandelbrot_escape(escape_data, xmin, xmax, ymin, ymax, max_iter)
