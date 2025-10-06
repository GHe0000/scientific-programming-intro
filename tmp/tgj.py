import numpy as np
import numba as nb

@nb.njit
def iterate_f(x0, r, n_iter):
    x = x0
    for _ in range(n_iter):
        x = r - x**2
    return x

@nb.njit
def find_bifurcation(n, r_left, r_right, tol=1e-7, max_iter=100):
    x0 = 0.0
    for _ in range(max_iter):
        r_mid = (r_left + r_right)/2
        n_iter = 1000 + 100*2**n  # 增加迭代次数保证轨道稳定
        x = iterate_f(x0, r_mid, n_iter)
        x_prev = iterate_f(x0, r_mid, n_iter//2)
        if abs(x - x_prev) < 1e-10:
            r_right = r_mid
        else:
            r_left = r_mid
        if r_right - r_left < tol:
            break
    return (r_left + r_right)/2

@nb.njit(parallel=True)
def compute_bifurcations(bounds):
    n_points = bounds.shape[0]
    r_n = np.zeros(n_points)
    for i in nb.prange(n_points):
        n = i+1
        r_n[i] = find_bifurcation(n, bounds[i,0], bounds[i,1])
    return r_n

# 搜索区间，可根据经验微调
r_bounds = np.array([
    [0.74, 0.76],
    [1.24, 1.26],
    [1.353, 1.355],
    [1.381, 1.383],
    [1.3875,1.389],
    [1.3895,1.3905],
])

r_n = compute_bifurcations(r_bounds)
print("分叉点 r_n:")
print(r_n)

# 计算 Feigenbaum 常数 δ_n
delta_n = [(r_n[i]-r_n[i-1])/(r_n[i+1]-r_n[i]) for i in range(1,len(r_n)-1)]
print("\nFeigenbaum 常数 δ_n:")
print(delta_n)
