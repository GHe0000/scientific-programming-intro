import numpy as np
from numba import njit, prange

@njit
def calc_cycle_single(r, x0=0.0, tol=1e-6, n_warm=1000, max_period=32):
    """
    检测给定 r 的周期数（最大为 max_period）
    超过 max_period 或无循环则返回 -1
    """
    # 预热迭代，使轨道进入吸引子附近
    x = x0
    for _ in range(n_warm):
        x = r - x * x

    # 保存轨道点（仅最多 max_period+1 个）
    xs = np.empty(max_period + 1)
    xs[0] = x
    n_stored = 1

    for n in range(1, max_period + 1):
        x = r - x * x
        # 检查是否与已有点接近
        for i in range(n_stored):
            if abs(x - xs[i]) < tol:
                return n - i  # 周期长度
        xs[n_stored] = x
        n_stored += 1

    # 超过 max_period 仍无重复，视为无周期
    return -1

@njit(parallel=True)
def detect_periods(r_arr, tol=1e-6, n_warm=1000, max_iter=2000, max_period=32):
    n = len(r_arr)
    periods = np.empty(n, dtype=np.int32)
    for i in prange(n):
        periods[i] = detect_period(r_arr[i], tol, n_warm, max_iter, max_period)
    return periods



r_arr = np.linspace(0, 1.41, 5000)
periods = detect_periods(r_arr, tol=1e-10, n_warm=1000, max_iter=2000, max_period=64)

# 绘制周期图
import matplotlib.pyplot as plt
mask = (periods > 0)
plt.plot(r_arr[mask], periods[mask], ".", ms=2)
plt.show()
