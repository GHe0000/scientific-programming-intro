import numpy as np
import numba as nb

# @nb.njit()
def calc_lyapunov(f, df, x0, r_arr, n_warm, n_sample):
    x0 = np.full_like(r_arr, x0)
    for _ in range(n_warm):
        x0 = f(r_arr, x0)
    sum_ln = np.zeros_like(r_arr)
    for _ in range(n_sample):
        x0 = f(r_arr, x0)
        sum_ln += np.log(np.abs(df(r_arr, x0)+1e-14)) # offset, 防止 log(0)
    return sum_ln / n_sample
