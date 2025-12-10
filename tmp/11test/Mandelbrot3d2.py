import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt

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

@nb.njit(parallel=True)
def compute_orbits_flat(c_real_flat, c_imag_flat, max_iter, skip_iter, keep_iter):
    n = len(c_real_flat)
    total_points = n * keep_iter
    
    out_x = np.empty(total_points, dtype=np.float32)
    out_y = np.empty(total_points, dtype=np.float32)
    out_z = np.empty(total_points, dtype=np.float32)
    
    for i in nb.prange(n):
        c = complex(c_real_flat[i], c_imag_flat[i])
        z = 0j
        
        # 1. 跑掉瞬态 (Transient)
        for _ in range(skip_iter):
            z = z*z + c
            
        # 2. 记录稳定轨道
        base_idx = i * keep_iter
        for k in range(keep_iter):
            z = z*z + c            
            # 存储数据
            out_x[base_idx + k] = c.real
            out_y[base_idx + k] = c.imag
            out_z[base_idx + k] = z.real 
    return out_x, out_y, out_z

t0 = time.time()
w, h = 800, 600
xmin, xmax = -2.0, 0.8
ymin, ymax = -1.1, 1.1

max_iter = 2000
skip_iter = 20000
keep_iter = 1000

mask = calc_mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter)
print(f"Done 1:{time.time()-t0 :.3f}s")

y_indices, x_indices = np.where(mask == 0)

dx = (xmax - xmin) / w
dy = (ymax - ymin) / h

c_reals = xmin + x_indices * dx
c_imags = ymin + y_indices * dy

total_interior = len(c_reals)
max_display_points = 1000

if total_interior > max_display_points:
    indices = np.random.choice(total_interior, max_display_points, replace=False)
    c_reals = c_reals[indices]
    c_imags = c_imags[indices]
xs, ys, zs = compute_orbits_flat(c_reals, c_imags, max_iter, skip_iter, keep_iter)
print(f"Done 2:{time.time()-t0 :.3f}s")

valid_mask = (zs >= -4) & (zs <= 4)
xs_plot = xs[valid_mask]
ys_plot = ys[valid_mask]
zs_plot = zs[valid_mask]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xs_plot, ys_plot, zs_plot, s=0.5, 
                     c=zs_plot, cmap='ocean', 
                     alpha=0.3, linewidth=0,
                     vmin=-5, vmax=5) # 显式设置颜色范围

ax.set_xlabel('Re(c)')
ax.set_ylabel('Im(c)')
ax.set_zlabel('Re(z)')
ax.set_title('Mandelbrot Set Bifurcation Diagram')

ax.view_init(elev=20, azim=-120)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Re(z)')
plt.savefig('save1.png')
print(f"Done 3:{time.time()-t0 :.3f}s")
plt.show()
print(f"Done:{time.time()-t0 :.3f}s")
