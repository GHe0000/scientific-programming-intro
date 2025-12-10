import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@nb.njit(parallel=True, cache=True)
def calc_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, tol, max_period):
    esc_map = np.zeros((height, width), dtype=np.int32)
    per_map = np.zeros((height, width), dtype=np.int32)
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    for y in nb.prange(height):
        for x in range(width):
            c = complex(xmin + x * dx, ymin + y * dy)
            z = 0j
            is_esc = False
            iter_count = 0
            for i in range(max_iter):
                if z.real*z.real + z.imag*z.imag > 4.0:
                    is_esc = True
                    iter_count = i
                    break
                z = z*z + c
            if is_esc:
                esc_map[y, x] = iter_count
                per_map[y, x] = 0 
            else:
                esc_map[y, x] = max_iter
                z0 = z  # 记录当前稳定状态
                found_p = -1    # 默认 -1 (代表未知周期或高阶周期)
                for p in range(1, max_period + 1):
                    z = z*z + c
                    if abs(z - z0) < tol:
                        found_p = p
                        break
                per_map[y, x] = found_p
    return esc_map, per_map

width, height = 1600, 1400
xmin, xmax = -2.0, 0.6
ymin, ymax = -1.2, 1.2
max_iter = 1000
tol = 1e-4
max_period = 12

esc_map, per_map = calc_mandelbrot(
    xmin, xmax, ymin, ymax, width, height, max_iter, tol, max_period
)

fig, ax = plt.subplots(figsize=(16, 11), dpi=120)

# -------------------------------------------------
# 图层 1 (底层): 外部逃逸时间 (Escape Velocity)
# -------------------------------------------------
mask_escape = np.ma.masked_equal(esc_map, max_iter)

# 外部色图: Magma (黑-紫-橙-黄)
cmap_esc = plt.cm.magma.copy()
cmap_esc.set_bad('black')

norm_esc = mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=max_iter)

im_escape = ax.imshow(mask_escape, extent=[xmin, xmax, ymin, ymax], 
                      cmap=cmap_esc, norm=norm_esc, origin='lower', interpolation='bilinear')

# -------------------------------------------------
# 图层 2 (顶层): 内部周期结构 (Periodicity)
# -------------------------------------------------
mask_period = np.ma.masked_equal(per_map, 0)

# 准备离散色图
# 我们需要 max_period + 1 种颜色 (索引 0 用于"其他/未知", 索引 1..N 用于周期)
# 使用 Tab20 色图，颜色鲜明
base_colors = plt.cm.tab20.colors 

# 构建颜色列表: 第一个颜色是灰色(对应未知周期 -1)，后面是 Tab20
color_list = ['dimgray'] + list(base_colors[:max_period])
cmap_period = mcolors.ListedColormap(color_list)

# 数据预处理: 将 -1 (未知周期) 映射为 0，以便配合 BoundaryNorm
viz_period_data = per_map.copy()
viz_period_data[viz_period_data == -1] = 0 

bounds = np.arange(-0.5, max_period + 1.5, 1)
norm_period = mcolors.BoundaryNorm(bounds, cmap_period.N)

im_period = ax.imshow(mask_period, extent=[xmin, xmax, ymin, ymax],
                      cmap=cmap_period, norm=norm_period, origin='lower', interpolation='nearest')

plt.subplots_adjust(right=0.82) 

cax1 = fig.add_axes([0.84, 0.15, 0.015, 0.7]) 
cbar1 = plt.colorbar(im_escape, cax=cax1)
cbar1.set_label("Escape Velocity", color='black', fontsize=10, labelpad=10)

cax2 = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
cbar2 = plt.colorbar(im_period, cax=cax2, ticks=np.arange(0, max_period + 1))
labels = ['Other'] + [str(i) for i in range(1, max_period + 1)]
cbar2.ax.set_yticklabels(labels)
cbar2.set_label("Period", color='black', fontsize=10, labelpad=10)

ax.set_title("Mandelbrot Set", fontsize=16, pad=15)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.show()
