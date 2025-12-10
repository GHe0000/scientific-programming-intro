import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@nb.njit(parallel=True, cache=True)
def calc_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0=0j):
    is_esc_map = np.zeros((height, width), dtype=np.int32)
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    for y in nb.prange(height):
        for x in range(width):
            c = complex(xmin + x * dx, ymin + y * dy)
            z = z0
            is_esc = False
            for i in range(max_iter):
                if z.real*z.real + z.imag*z.imag > 4.0:
                    is_esc = True
                    break
                z = z*z + c
            if is_esc:
                is_esc_map[y, x] = 1
    return is_esc_map

# --- 2. 核心修改：计算 |z| 和 相位 ---
@nb.njit(parallel=True)
def compute_orbits_flat(c_real_flat, c_imag_flat, max_iter, skip_iter, keep_iter):
    n = len(c_real_flat)
    total_points = n * keep_iter
    
    # 坐标数组
    out_x = np.empty(total_points, dtype=np.float32)
    out_y = np.empty(total_points, dtype=np.float32)
    
    # 修改点 1: 存储模长 |z| 作为高度
    out_z_abs = np.empty(total_points, dtype=np.float32)
    
    # 修改点 2: 存储相位 arg(z) 作为颜色
    out_phase = np.empty(total_points, dtype=np.float32)
    
    for i in nb.prange(n):
        c = complex(c_real_flat[i], c_imag_flat[i])
        z = 0j
        
        # 1. 跑掉瞬态
        for _ in range(skip_iter):
            z = z*z + c
            
        # 2. 记录稳定轨道
        base_idx = i * keep_iter
        for k in range(keep_iter):
            z = z*z + c
            
            out_x[base_idx + k] = c.real
            out_y[base_idx + k] = c.imag
            
            # 计算模长 |z|
            out_z_abs[base_idx + k] = np.abs(z)
            
            # 计算相位 (-pi 到 pi)
            out_phase[base_idx + k] = np.angle(z)
            
    return out_x, out_y, out_z_abs, out_phase

def main():
    # --- 参数设置 ---
    w, h = 800, 600
    xmin, xmax = -2.0, 0.8
    ymin, ymax = -1.1, 1.1
    
    max_iter = 2000
    skip_iter = 1000
    keep_iter = 30
    
    print(f"1. 计算 2D Mandelbrot 掩模 ({w}x{h})...")
    mask = calc_mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter)
    
    # --- 数据准备 ---
    print("2. 提取内部点坐标...")
    y_indices, x_indices = np.where(mask == 0)
    dx = (xmax - xmin) / w
    dy = (ymax - ymin) / h
    c_reals = xmin + x_indices * dx
    c_imags = ymin + y_indices * dy
    
    # 随机降采样 (为了 Matplotlib 流畅度)
    total_interior = len(c_reals)
    max_display_points = 100000 
    
    if total_interior > max_display_points:
        print(f"   (降采样: {total_interior} -> {max_display_points})")
        indices = np.random.choice(total_interior, max_display_points, replace=False)
        c_reals = c_reals[indices]
        c_imags = c_imags[indices]

    print("3. Numba 加速计算轨道 (返回 |z| 和 相位)...")
    # 获取四个返回值
    xs, ys, z_abs, phases = compute_orbits_flat(c_reals, c_imags, max_iter, skip_iter, keep_iter)
    
    # --- Matplotlib 绘图 ---
    print("4. 绘制 3D 图 (Z=|z|, Color=Phase)...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 修改点 3: 
    # - z轴使用 z_abs
    # - c (颜色) 使用 phases
    # - cmap 使用 'hsv' (因为相位是循环的，hsv 首尾相接，-pi和pi颜色相同，非常适合相位图)
    scatter = ax.scatter(xs, ys, z_abs, s=0.5, c=phases, cmap='hsv', alpha=0.4, linewidth=0)
    
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    ax.set_zlabel('|z| (Modulus)') # 标签改为模长
    ax.set_title('Mandelbrot Orbits: Height=|z|, Color=Phase')
    
    # 视角调整
    ax.view_init(elev=30, azim=-120)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, 2.0) # |z| 对于内部点通常在 [0, 2] 之间
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Phase (radians)')
    # 设置刻度显示为 pi 符号
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    
    print("完成。")
    plt.show()

if __name__ == "__main__":
    main()
