import numpy as np
import numba as nb
import matplotlib.pyplot as plt
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

traj = lorenz_sample(skip_n=50000, save_n=int(2e6), per_n=1)
n_arr = np.unique(np.logspace(0.5, 2.0, num=10, dtype=int))

res1 = calc_box_counts(traj[:10], n_arr)

t0 = time.time()
res1 = calc_box_counts(traj, n_arr)
print(f"calc_box_counts: {time.time() - t0:.3f}s")

t0 = time.time()
res2 = calc_box_counts_use_histogramdd(traj, n_arr)
print(f"calc_box_counts_use_histogramdd: {time.time() - t0:.3f}s")

plt.loglog(n_arr, res1, "b-x", label='calc_box_counts')
plt.loglog(n_arr, res2, "r--+", label='calc_box_counts_use_histogramdd')
plt.legend()
plt.show()
