
import numpy as np
import numba as nb

import matplotlib.pyplot as plt

import time

@nb.njit(cache=True)
def metropolis_samples(T, k, alpha, beta, n_samples, step_size=1.0):
    V = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    samples = np.empty(n_samples)
    beta_T = 1.0 / (1.0*T) # kb = 1
    x = 0.0
    V_old = V(x)
    for i in range(n_samples):
        x_new = x + np.random.uniform(-step_size, step_size)
        V_new = V(x_new)
        dV = V_new - V_old
        accept = True if dV <= 0 else (np.random.random() < np.exp(-beta_T * dV))
        if accept:
            x = x_new
            V_old = V_new
        samples[i] = x
    return samples

@nb.njit(cache=True, parallel=True)
def samples_T_arr(T_arr, k, alpha, beta, n_samples, step_size=1.0):
    n_T = len(T_arr)
    ret = np.empty((n_T, n_samples))
    for i in nb.prange(n_T):
        ret[i] = metropolis_samples(T_arr[i], k, alpha, beta, n_samples, step_size)
    return ret

T_arr = np.linspace(0.001, 2, 100)
n_samples = int(1e6)
k = 1.0
alpha = 0.0
beta = 0.0

print("Build...")
_ = samples_T_arr(T_arr[:2], k, alpha, beta, n_samples)

print("Run...")
t_start = time.time()
samples_arr = samples_T_arr(T_arr, k, alpha, beta, n_samples)
print(f"Time: {time.time() - t_start:.3f}")

x_avg = np.mean(samples_arr, axis=1)

plt.plot(T_arr, x_avg)
plt.show()

