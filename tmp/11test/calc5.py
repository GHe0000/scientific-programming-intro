import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
import time

@nb.njit(cache=True)
def lorenz_sample(skip_n, save_n, per_n, dt=0.01):
    def lorenz(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])
    def rk4_step(state, dt):
        k1 = lorenz(state)
        k2 = lorenz(state + k1 * dt * 0.5)
        k3 = lorenz(state + k2 * dt * 0.5)
        k4 = lorenz(state + k3 * dt)
        return state + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
    X = np.array([0.1, 0.0, 0.0])
    for _ in range(skip_n):
        X = rk4_step(X, dt)
    ret = np.empty((save_n, 3), dtype=np.float64)
    for i in range(save_n):
        ret[i] = X
        for _ in range(per_n):
            X = rk4_step(X, dt)
    return ret

def calc_box_counts_use_histogramdd(traj, n_arr):
    num_n = len(n_arr)
    res = np.empty(num_n, dtype=np.int64)

    min_vals = np.min(traj, axis=0)
    max_vals = np.max(traj, axis=0)
    margin = (max_vals - min_vals) * 1e-5
    min_vals -= margin
    max_vals += margin
    bins_range = np.stack((min_vals, max_vals), axis=1)

    for i, n in enumerate(n_arr):
        n = int(n)
        H, _ = np.histogramdd(traj, bins=n, range=bins_range)
        res[i] = np.count_nonzero(H)
    return res

@nb.njit(parallel=True, cache=True)
def calc_box_counts(traj, n_arr):
    n_pts = len(traj)
    num_n = len(n_arr)
    res = np.empty(num_n, dtype=np.int64)

    min_vals = np.empty(3, dtype=np.float64)
    max_vals = np.empty(3, dtype=np.float64)
    for k in range(3):
        min_vals[k] = np.min(traj[:, k])
        max_vals[k] = np.max(traj[:, k])
    margin = (max_vals - min_vals) * 1e-5
    min_vals -= margin
    max_vals += margin
    
    # 并行计算
    for i in nb.prange(num_n):
        n = int(n_arr[i])
        dx = (max_vals[0] - min_vals[0]) / n
        dy = (max_vals[1] - min_vals[1]) / n
        dz = (max_vals[2] - min_vals[2]) / n
        hashes = np.empty(n_pts, dtype=np.int64)

        for p_idx in range(n_pts):
            ix = int((traj[p_idx, 0] - min_vals[0]) / dx)
            iy = int((traj[p_idx, 1] - min_vals[1]) / dy)
            iz = int((traj[p_idx, 2] - min_vals[2]) / dz)
            if ix >= n: ix = n - 1
            if iy >= n: iy = n - 1
            if iz >= n: iz = n - 1
            hashes[p_idx] = ix + iy * n + iz * n * n
        hashes.sort()
        
        n_boxes = 0
        if n_pts > 0:
            n_boxes = 1
            for k in range(1, n_pts):
                if hashes[k] != hashes[k-1]:
                    n_boxes += 1
        res[i] = n_boxes
    return res

@nb.njit(parallel=True, cache=True)
def calc_entropies(traj, n_arr):
    n_pts = len(traj)
    num_n = len(n_arr)
    res = np.empty(num_n, dtype=np.float64)

    min_vals = np.empty(3, dtype=np.float64)
    max_vals = np.empty(3, dtype=np.float64)
    for k in range(3):
        min_vals[k] = np.min(traj[:, k])
        max_vals[k] = np.max(traj[:, k])
    margin = (max_vals - min_vals) * 1e-5
    min_vals -= margin
    max_vals += margin
    
    # 并行计算
    for i in nb.prange(num_n):
        n = int(n_arr[i])
        dx = (max_vals[0] - min_vals[0]) / n
        dy = (max_vals[1] - min_vals[1]) / n
        dz = (max_vals[2] - min_vals[2]) / n
        hashes = np.empty(n_pts, dtype=np.int64)
        
        for p_idx in range(n_pts):
            ix = int((traj[p_idx, 0] - min_vals[0]) / dx)
            iy = int((traj[p_idx, 1] - min_vals[1]) / dy)
            iz = int((traj[p_idx, 2] - min_vals[2]) / dz)
            if ix >= n: ix = n - 1
            if iy >= n: iy = n - 1
            if iz >= n: iz = n - 1
            hashes[p_idx] = ix + iy * n + iz * n * n
        hashes.sort()
        
        # 计算熵
        entropy = 0.0
        current_run_count = 1
        for k in range(1, n_pts):
            if hashes[k] == hashes[k-1]:
                current_run_count += 1
            else:
                p = current_run_count / n_pts
                entropy -= p * np.log(p)
                current_run_count = 1
        p = current_run_count / n_pts
        entropy -= p * np.log(p)
        res[i] = entropy
    return res

def calc_correlation_sums(traj, r_arr):
    n_pts = len(traj)
    tree = cKDTree(traj)
    counts = tree.count_neighbors(tree, r_arr, cumulative=False)
    c_r = (counts.astype(np.float64) - n_pts) / (n_pts * (n_pts - 1))
    return c_r

def fit_and_plot_ransac(ax, x, y):
    X = x.reshape(-1, 1)
    Y = y
    ransac = RANSACRegressor(min_samples=0.4, 
                             residual_threshold=0.05,
                             random_state=42)
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)
    slope = ransac.estimator_.coef_[0]
    ax.scatter(x[outlier_mask], y[outlier_mask], color='r', marker='x', s=15, label='Outliers')
    ax.scatter(x[inlier_mask], y[inlier_mask], color='b', marker='o', s=15, label='Inliers')
    ax.plot(line_x, line_y, "k--" ,label=f'Slope = {slope:.4f}')
    return slope

save_n = int(1e8)  # 1亿点
print(f"[{time.strftime('%H:%M:%S')}] 正在生成 Lorenz 轨迹 ({save_n} 点)...")
traj = lorenz_sample(skip_n=50000, save_n=save_n, per_n=1)
print(f"[{time.strftime('%H:%M:%S')}] 轨迹生成完毕。")
n_arr = np.unique(np.logspace(0.5, 4.5, num=50, dtype=int))
print(f"[{time.strftime('%H:%M:%S')}] 正在并行计算 D0 (Box Counts)...")
t0 = time.time()
box_counts = calc_box_counts(traj, n_arr)
print(f"[{time.strftime('%H:%M:%S')}] D0 计算完成 ({time.time()-t0:.2f}s)。")
print(f"[{time.strftime('%H:%M:%S')}] 正在并行计算 D1 (Entropies)...")
t0 = time.time()
entropies = calc_entropies(traj, n_arr)
print(f"[{time.strftime('%H:%M:%S')}] D1 计算完成 ({time.time()-t0:.2f}s)。")
d2_sample_size = 500000 
print(f"[{time.strftime('%H:%M:%S')}] 正在为 D2 降采样至 {d2_sample_size} 点...")
indices = np.random.choice(len(traj), d2_sample_size, replace=False)
traj_subset = traj[indices]

r_arr = np.logspace(-1.5, 1.5, 60)

print(f"[{time.strftime('%H:%M:%S')}] 正在计算 D2 (Correlation Sums)...")
t0 = time.time()
c_r_values = calc_correlation_sums(traj_subset, r_arr)
print(f"[{time.strftime('%H:%M:%S')}] D2 计算完成 ({time.time()-t0:.2f}s)。")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

d0 = fit_and_plot_ransac(axes[0], 
                         np.log(n_arr), 
                         np.log(box_counts)) 

d1 = fit_and_plot_ransac(axes[1], 
                         np.log(n_arr), 
                         entropies) 

valid_mask = (r_arr > 0) & (c_r_values > 0)
r_arr = r_arr[valid_mask]
c_r_values = c_r_values[valid_mask]
d2 = fit_and_plot_ransac(axes[2], 
                         np.log(r_arr), 
                         np.log(c_r_values)) 

print("-" * 40)
print(f"容量维数 D0: {d0:.4f}")
print(f"信息维数 D1: {d1:.4f}")
print(f"关联维数 D2: {d2:.4f}")
print("-" * 40)

plt.tight_layout()
plt.show()
