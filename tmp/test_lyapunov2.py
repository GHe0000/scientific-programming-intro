import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from timer import FunctionTimer

@FunctionTimer
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

'''
__________ TEST CODE _________
'''

f = lambda r,x : r*x*(1-x)
df = lambda r,x : r*(1-2*x)

# f = lambda r,x : r*x
# df = lambda r,x : r
x0 = 0.3
r_arr = np.linspace(0, 4, 1000)
f = nb.njit(f, inline='always')
df = nb.njit(df, inline='always')
ly_arr = calc_lyapunov(f, df, x0, r_arr, 1000, 100000)
plt.plot(r_arr, ly_arr)
plt.show()

