import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from sklearn.linear_model import RANSACRegressor
import time

# def correlation_integral(pts, r_list):
#     N = pts.shape[0]
#     sum_sq = np.sum(pts**2, axis=1)
#     dists_sq = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * np.dot(pts, pts.T)
#     dists_sq = np.maximum(dists_sq, 0)
#     dists_arr = np.sqrt(dists_sq[np.triu_indices(N, k=1)])
#     dists_arr.sort()
#     count = np.searchsorted(dists_arr, r_list, side='right')
#     count_tot = 2 * count + N
#     C_r = count_tot / (N ** 2)
#     return C_r

def correlation_integral(pts, r_list, nbins=2000):
    N = pts.shape[0]
    r_max = r_list.max()
    hist = np.zeros(nbins, dtype=np.int64)
    bin_edges = np.linspace(0, r_max, nbins+1)
    for i in range(N):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        h, _ = np.histogram(d, bins=bin_edges)
        hist += h
    cum_hist = np.cumsum(hist)
    C_r = np.interp(r_list, bin_edges[1:], cum_hist) * 2 / N**2
    return C_r

def correlation_integral_rlist(pts, r_list):
    N = pts.shape[0]
    L = len(r_list)
    r_list = np.asarray(r_list)
    cum_counts = np.zeros(L, dtype=np.int64)
    for i in range(N):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        d.sort()
        cum_counts += np.searchsorted(d, r_list, side='right')
    C_r = (2 * cum_counts + N) / (N ** 2)
    return C_r

@nb.njit(fastmath=True, cache=True)
def correlation_integral_numba(pts, r_list):
    N, M = pts.shape
    L = len(r_list)
    cum_counts = np.zeros(L, dtype=np.int64)
    r_list_sorted = np.sort(r_list)
    for i in range(N):
        count_vec = np.zeros(L, dtype=np.int64)
        for j in range(i+1, N):
            # ||x_i - x_j||
            s = 0.0
            for k in range(M):
                diff = pts[i, k] - pts[j, k]
                s += diff * diff
            d = np.sqrt(s)
            lo = 0
            hi = L
            while lo < hi:
                mid = (lo + hi) // 2
                if r_list_sorted[mid] < d:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < L:
                count_vec[lo] += 1
        for kk in range(L):
            cum_counts[kk] += count_vec[kk]
    for kk in range(1, L):
        cum_counts[kk] += cum_counts[kk-1]
    return (2*cum_counts + N) / (N*N)

def circle(n=20000):
    theta = np.linspace(0, 2*np.pi, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])

def sierpinski_carpet(n=100000): # log(8)/log(3)=1.8927
    p = np.random.rand(2)
    pts = []
    for _ in range(n):
        i, j = divmod(np.random.randint(0,8), 3)
        if (i,j) == (1,1):
            continue
        p = (p + np.array([i,j])) / 3
        pts.append(p)
    return np.array(pts)

M = 1
N = 20000

# pts = np.random.rand(N, M)
# pts = circle(N)
pts = sierpinski_carpet(int(1e5))
idx_choice = np.random.choice(pts.shape[0], size=N, replace=False)
pts = pts[idx_choice]

plt.plot(pts[:,0], pts[:,1],"k.")
plt.axis("equal")
plt.show()

r_min, r_max = 0.1, 1.0
r_arr = np.logspace(np.log(r_min), np.log(r_max), 100)
t1 = time.time()
C_r = correlation_integral(pts, r_arr)
print(f"{time.time() - t1:.3f}")
t1 = time.time()
C_r = correlation_integral_numba(pts, r_arr)
print(f"{time.time() - t1:.3f}")

ln_r = np.log(r_arr)
ln_C_r = np.log(C_r)
ransac = RANSACRegressor(min_samples=2, residual_threshold=0.01, random_state=42)
ransac.fit(ln_r.reshape(-1, 1), ln_C_r.reshape(-1, 1))

k_fit = ransac.estimator_.coef_[0][0]
b_fit = ransac.estimator_.intercept_[0]

print(f"k_fit = {k_fit:.3f}, b_fit = {b_fit:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(ln_r,ln_C_r,"k--.",label="Data")
plt.plot(ln_r, k_fit * ln_r + b_fit, color='red',label=f"Fit {k_fit:.3f}")
plt.plot(ln_r, M*ln_r+b_fit, color='blue',label=f"k={M}")
plt.xlabel("ln(r)")
plt.ylabel("ln(Cr)")
plt.legend()
plt.grid()
plt.title("Random")
plt.show()


# xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
# grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
# pts = grid_pts
# r_min, r_max = 0.2, 1.2
# r_arr = np.logspace(np.log(r_min), np.log(r_max), 100)
# C_r = correlation_integral(pts, r_arr)
#
# ln_r = np.log(r_arr)
# ln_C_r = np.log(C_r)
# ransac = RANSACRegressor(min_samples=2, residual_threshold=0.05, random_state=42)
# ransac.fit(ln_r.reshape(-1, 1), ln_C_r.reshape(-1, 1))
# print(f"k_fit = {k_fit:.3f}, b_fit = {b_fit:.3f}")
#
# plt.figure(figsize=(8, 6))
# plt.plot(ln_r,ln_C_r,"k--.",label="Data")
# plt.plot(ln_r, k_fit * ln_r + b_fit, color='red',label=f"Fit {k_fit:.3f}")
# plt.plot(ln_r, M*ln_r+b_fit, color='blue',label=f"k={M}")
# plt.xlabel("ln(r)")
# plt.ylabel("ln(Cr)")
# plt.legend()
# plt.title("Grid")
# plt.grid()
# plt.show()
#
#
# xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
# grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
# idx_choice = np.random.choice(grid_pts.shape[0], size=2500, replace=False)
# pts = grid_pts[idx_choice]
# r_min, r_max = 0.1, 1.2
# r_arr = np.logspace(np.log(r_min), np.log(r_max), 100)
# C_r = correlation_integral(pts, r_arr)
#
# ln_r = np.log(r_arr)
# ln_C_r = np.log(C_r)
#
# k_fit, b_fit = np.polyfit(ln_r, ln_C_r, 1)
# print(f"k_fit = {k_fit:.3f}, b_fit = {b_fit:.3f}")
#
# plt.figure(figsize=(8, 6))
# plt.plot(ln_r,ln_C_r,"k--.",label="Data")
# plt.plot(ln_r, k_fit * ln_r + b_fit, color='red',label=f"Fit {k_fit:.3f}")
# plt.plot(ln_r, M*ln_r+b_fit, color='blue',label=f"k={M}")
# plt.xlabel("ln(r)")
# plt.ylabel("ln(Cr)")
# plt.legend()
# plt.title("Grid")
# plt.grid()
# plt.show()
