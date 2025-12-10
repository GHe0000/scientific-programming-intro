import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit(parallel=True, cache=True)
def calc_mandelbrot_range(xmin, xmax, ymin, ymax, width, height, max_iter, z0=0j):
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
def calc_pt(c_real_flat, c_imag_flat, skip_iter, keep_iter):
    n = len(c_real_flat)
    total_points = n * keep_iter
    out_x = np.empty(total_points, dtype=np.float32)
    out_y = np.empty(total_points, dtype=np.float32)
    out_z_abs = np.empty(total_points, dtype=np.float32)
    out_phase = np.empty(total_points, dtype=np.float32)
    for i in nb.prange(n):
        c = complex(c_real_flat[i], c_imag_flat[i])
        z = 0j
        for _ in range(skip_iter):
            z = z*z + c
        base_idx = i * keep_iter
        for k in range(keep_iter):
            z = z*z + c
            out_x[base_idx + k] = c.real
            out_y[base_idx + k] = c.imag
            out_z_abs[base_idx + k] = np.abs(z)
            out_phase[base_idx + k] = np.angle(z)
    return out_x, out_y, out_z_abs, out_phase


@nb.njit()
def voxelize_points(xs, ys, zs, phases, bounds, grid_dims):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = grid_dims
    count_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    phase_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    
    n_points = len(xs)
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    dz = (zmax - zmin)
    
    for i in range(n_points):
        x, y, z, p = xs[i], ys[i], zs[i], phases[i]
        if not (xmin <= x < xmax and ymin <= y < ymax and zmin <= z < zmax):
            continue
        idx_x = int((x - xmin) / dx * nx)
        idx_y = int((y - ymin) / dy * ny)
        idx_z = int((z - zmin) / dz * nz)
        if idx_x >= nx: idx_x = nx - 1
        if idx_y >= ny: idx_y = ny - 1
        if idx_z >= nz: idx_z = nz - 1
        
        count_grid[idx_x, idx_y, idx_z] += 1
        phase_grid[idx_x, idx_y, idx_z] += p

    valid_count = 0
    flat_len = nx * ny * nz
    for i in range(flat_len):
        if count_grid.flat[i] > 0:
            valid_count += 1
    out_vx = np.empty(valid_count, dtype=np.float32)
    out_vy = np.empty(valid_count, dtype=np.float32)
    out_vz = np.empty(valid_count, dtype=np.float32)
    out_vphase = np.empty(valid_count, dtype=np.float32)
    
    ptr = 0
    for ix in range(nx):
        center_x = xmin + (ix + 0.5) * (dx / nx)
        for iy in range(ny):
            center_y = ymin + (iy + 0.5) * (dy / ny)
            for iz in range(nz):
                c = count_grid[ix, iy, iz]
                if c > 0:
                    center_z = zmin + (iz + 0.5) * (dz / nz)
                    
                    out_vx[ptr] = center_x
                    out_vy[ptr] = center_y
                    out_vz[ptr] = center_z
                    out_vphase[ptr] = phase_grid[ix, iy, iz] / c # 平均相位
                    ptr += 1
                    
    return out_vx, out_vy, out_vz, out_vphase

w, h = 800, 600
xmin, xmax = -2.0, 0.8
ymin, ymax = -1.1, 1.1

zmin, zmax = 0.0, 2.0 

max_iter = 5000
skip_iter = 2000
keep_iter = 1000 
voxel_res = (500, 500, 500) 

mask = calc_mandelbrot_range(xmin, xmax, ymin, ymax, w, h, max_iter)

# 提取内部点
y_indices, x_indices = np.where(mask == 0)
dx_map = (xmax - xmin) / w
dy_map = (ymax - ymin) / h
c_reals = xmin + x_indices * dx_map
c_imags = ymin + y_indices * dy_map

total_interior = len(c_reals)

print("2. Calculating Orbits...")
raw_xs, raw_ys, raw_zs, raw_phases = calc_pt(c_reals, c_imags, skip_iter, keep_iter)

print(f"3. Voxelizing into grid {voxel_res}...")
bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
vox_x, vox_y, vox_z, vox_phase = voxelize_points(raw_xs, raw_ys, raw_zs, raw_phases, bounds, voxel_res)

print(f"   Reduced points to draw: {len(vox_x)} (Optimized!)")

print("4. Plotting...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

marker_size = 2.0 
scatter = ax.scatter(vox_x, vox_y, vox_z, s=marker_size, c=vox_phase, 
                     cmap='hsv', alpha=0.6, linewidth=0, marker='s') # marker='s' 方块看起来更像体素

ax.set_xlabel('Re(c)')
ax.set_ylabel('Im(c)')
ax.set_zlabel('|z|')
ax.set_title(f'Mandelbrot Orbits (Voxelized)\nGrid: {voxel_res} | Points: {len(vox_x)}')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.view_init(elev=30, azim=-120)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Average Phase')
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

plt.tight_layout()
plt.show()
