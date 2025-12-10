import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit(parallel=True, cache=True)
def calc_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0=0j):
    is_esc_map = np.zeros((height, width), dtype=np.int32)
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    for y in nb.prange(height):
        c_imag = ymin + y * dy
        for x in range(width):
            c_real = xmin + x * dx
            c = complex(c_real, c_imag)
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
def calc_bifurcation_pt(c_real, c_imag, skip_n, save_n):
    n = len(c_real)
    tot_pt = n * save_n
    
    out_x = np.empty(tot_pt, dtype=np.float32)
    out_y = np.empty(tot_pt, dtype=np.float32)
    out_z_real = np.empty(tot_pt, dtype=np.float32)
    for i in nb.prange(n):
        c = complex(c_real[i], c_imag[i])
        z = 0j
        for _ in range(skip_n):
            z = z*z + c
        base_idx = i * save_n
        for k in range(save_n):
            z = z*z + c
            out_x[base_idx + k] = c.real
            out_y[base_idx + k] = c.imag
            out_z_real[base_idx + k] = z.real
    return out_x, out_y, out_z_real

@nb.njit(parallel=False)
def vox_pt(xs, ys, zs, values, bounds, grid_dims):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = grid_dims
    
    count_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    value_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    
    n_points = len(xs)
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    dz = (zmax - zmin)
    
    for i in range(n_points):
        x, y, z, v = xs[i], ys[i], zs[i], values[i]
        
        if not (xmin <= x < xmax and ymin <= y < ymax and zmin <= z < zmax):
            continue
            
        idx_x = int((x - xmin) / dx * nx)
        idx_y = int((y - ymin) / dy * ny)
        idx_z = int((z - zmin) / dz * nz)
        
        if idx_x >= nx: idx_x = nx - 1
        if idx_y >= ny: idx_y = ny - 1
        if idx_z >= nz: idx_z = nz - 1
        
        count_grid[idx_x, idx_y, idx_z] += 1
        value_grid[idx_x, idx_y, idx_z] += v

    valid_count = 0
    flat_len = nx * ny * nz
    for i in range(flat_len):
        if count_grid.flat[i] > 0:
            valid_count += 1
            
    out_vx = np.empty(valid_count, dtype=np.float32)
    out_vy = np.empty(valid_count, dtype=np.float32)
    out_vz = np.empty(valid_count, dtype=np.float32)
    out_vcolor = np.empty(valid_count, dtype=np.float32)
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
                    out_vcolor[ptr] = value_grid[ix, iy, iz] / c 
                    ptr += 1
    return out_vx, out_vy, out_vz, out_vcolor

w, h = 800, 600
xmin, xmax = -2.0, 0.8
ymin, ymax = -1.1, 1.1

zmin, zmax = -2.0, 2.0 

max_iter = 5000
skip_iter = 2000
keep_iter = 1000 
vox_res = (500, 500, 500) 

mask = calc_mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter)
y_in_arr, x_in_arr = np.where(mask == 0)
dx_map = (xmax - xmin) / w
dy_map = (ymax - ymin) / h
c_reals = xmin + x_in_arr * dx_map
c_imags = ymin + y_in_arr * dy_map

raw_xs, raw_ys, raw_z_real = calc_bifurcation_pt(c_reals, c_imags, skip_iter, keep_iter)

bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
vox_x, vox_y, vox_z, vox_c = vox_pt(raw_xs, raw_ys, raw_z_real, raw_z_real, bounds, vox_res)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(vox_x, vox_y, vox_z, s=2.0, c=vox_c, 
                     cmap='viridis', alpha=0.3, linewidth=0, marker='s')

ax.set_xlabel('Re(c)')
ax.set_ylabel('Im(c)')
ax.set_zlabel('Re(z)') 
ax.set_title('Mandelbrot Orbits: Height & Color = Re(z)')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.view_init(elev=30, azim=-120)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Re(z) Value')

plt.tight_layout()
plt.show()
