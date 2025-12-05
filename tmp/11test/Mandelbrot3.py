import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit, prange

@njit(parallel=True)
def compute_mandelbrot_layers(xmin, xmax, ymin, ymax, width, height, max_iter, tol, max_period):
    escape_map = np.zeros((height, width), dtype=np.int32)
    period_map = np.zeros((height, width), dtype=np.int32)
    
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
                if z.real*z.real + z.imag*z.imag > 4.0:
                    escaped = True
                    iter_count = i
                    break
                z = z*z + c
            
            if escaped:
                escape_map[y, x] = iter_count
                period_map[y, x] = 0 
            else:
                escape_map[y, x] = max_iter
                
                z_snapshot = z  # 记录当前稳定状态
                found_p = -1    # 默认 -1 (代表未知周期或高阶周期)
                for p in range(1, max_period + 1):
                    z = z*z + c
                    if abs(z - z_snapshot) < tol:
                        found_p = p
                        break
                period_map[y, x] = found_p
    return escape_map, period_map

def plot_mandelbrot_complete(escape_data, period_data, xmin, xmax, ymin, ymax, max_iter, max_period):
    fig, ax = plt.subplots(figsize=(16, 11), dpi=120)
    
    # -------------------------------------------------
    # 图层 1 (底层): 外部逃逸时间 (Escape Velocity)
    # -------------------------------------------------
    # 掩盖掉内部点 (值为 max_iter 的点)
    mask_escape = np.ma.masked_equal(escape_data, max_iter)
    
    # 外部色图: Magma (黑-紫-橙-黄)
    cmap_esc = plt.cm.magma.copy()
    cmap_esc.set_bad('black') # 掩盖区域设为黑色(虽然会被上层覆盖)
    
    # 归一化: 使用 PowerNorm (gamma < 1) 提亮外部的深色区域，增加细节
    norm_esc = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=max_iter)
    
    im_escape = ax.imshow(mask_escape, extent=[xmin, xmax, ymin, ymax], 
                          cmap=cmap_esc, norm=norm_esc, origin='lower', interpolation='bilinear')
    
    # -------------------------------------------------
    # 图层 2 (顶层): 内部周期结构 (Periodicity)
    # -------------------------------------------------
    # 掩盖掉外部点 (值为 0 的点)
    mask_period = np.ma.masked_equal(period_data, 0)
    
    # 准备离散色图
    # 我们需要 max_period + 1 种颜色 (索引 0 用于"其他/未知", 索引 1..N 用于周期)
    # 使用 Tab20 色图，颜色鲜明
    base_colors = plt.cm.tab20.colors 
    
    # 构建颜色列表: 第一个颜色是灰色(对应未知周期 -1)，后面是 Tab20
    color_list = ['dimgray'] + list(base_colors[:max_period])
    cmap_period = mcolors.ListedColormap(color_list)
    
    # 数据预处理: 将 -1 (未知周期) 映射为 0，以便配合 BoundaryNorm
    viz_period_data = period_data.copy()
    viz_period_data[viz_period_data == -1] = 0 
    
    # 定义边界: [ -0.5, 0.5, 1.5, ... ] 确保整数落在色块中心
    bounds = np.arange(-0.5, max_period + 1.5, 1)
    norm_period = mcolors.BoundaryNorm(bounds, cmap_period.N)
    
    im_period = ax.imshow(mask_period, extent=[xmin, xmax, ymin, ymax],
                          cmap=cmap_period, norm=norm_period, origin='lower', interpolation='nearest')

    # -------------------------------------------------
    # 布局调整与双 Colorbar (左右堆叠)
    # -------------------------------------------------
    
    # 调整主图区域，右侧留出 20% 空间
    plt.subplots_adjust(right=0.82) 

    # --- 左侧 Colorbar: 逃逸速度 ---
    # 坐标: [left, bottom, width, height]
    cax1 = fig.add_axes([0.84, 0.15, 0.015, 0.7]) 
    cbar1 = plt.colorbar(im_escape, cax=cax1)
    cbar1.set_label('Escape Velocity (Iterations)', color='black', fontsize=10, labelpad=10)
    
    # --- 右侧 Colorbar: 内部周期 ---
    # 坐标: [left, bottom, width, height] -> 放在左侧条的右边
    cax2 = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
    cbar2 = plt.colorbar(im_period, cax=cax2, ticks=np.arange(0, max_period + 1))
    
    # 自定义刻度标签
    labels = ['Other'] + [str(i) for i in range(1, max_period + 1)]
    cbar2.ax.set_yticklabels(labels)
    cbar2.set_label('Internal Periodicity', color='black', fontsize=10, labelpad=10)

    # -------------------------------------------------
    # 标题与标签
    # -------------------------------------------------
    ax.set_title(f"Mandelbrot Analysis: Escape Time & Periodicity (1-{max_period})", fontsize=16, pad=15)
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    
    output_filename = 'mandelbrot_complete_analysis.png'
    plt.savefig(output_filename)
    print(f"渲染完成！图像已保存为: {output_filename}")
    plt.show()

# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == "__main__":
    # --- 参数设置 ---
    width, height = 1600, 1400  # 分辨率
    xmin, xmax = -2.0, 0.6      # 实部范围
    ymin, ymax = -1.2, 1.2      # 虚部范围
    
    max_iter = 500              # 逃逸检测最大迭代次数 (精度)
    tol = 1e-4                  # 周期检测的浮点容差
    max_period = 12             # 只区分并显示前 12 个周期，其他的显示为灰色
    
    print(f"正在计算 Mandelbrot 集合 (分辨率: {width}x{height}, 迭代: {max_iter})...")
    print("使用 Numba JIT 并行加速中...")
    
    # 1. 计算
    esc_map, per_map = compute_mandelbrot_layers(
        xmin, xmax, ymin, ymax, width, height, max_iter, tol, max_period
    )
    
    print("计算完成，正在生成可视化图像...")
    
    # 2. 绘图
    plot_mandelbrot_complete(
        esc_map, per_map, xmin, xmax, ymin, ymax, max_iter, max_period
    )
