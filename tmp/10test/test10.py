import numpy as np
import numba as nb

import time
import matplotlib.pyplot as plt

@nb.njit(cache=True)
def metropolis_samples(T, k, alpha, beta, n_samples, step_size=1.0):
    Vx = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    V = lambda x1,x2: Vx(x1) + Vx(-x2) + Vx(x2-x1)
    E = lambda X: V(X[0], X[1])
    samples = np.empty((n_samples, 2))
    beta_T = 1.0 / (1.0*T) # kb = 1
    X = np.zeros(2)
    E_old = E(X)
    for i in range(n_samples):
        X_new = X + np.random.uniform(-step_size, step_size, size=2)
        E_new = E(X_new)
        dE = E_new - E_old
        accept = True if dE <= 0 else (np.random.random() < np.exp(-beta_T * dE))
        if accept:
            X = X_new
            E_old = E_new
        samples[i] = X
    return samples

@nb.njit(cache=True, parallel=True)
def calc1(T_arr, k, alpha, beta, n_samples, calc_func ,step_size=1.0):
    n_T = len(T_arr)
    ret = np.zeros((n_T, 2))
    for i in nb.prange(n_T):
        samples = metropolis_samples(T_arr[i], k, alpha, beta, n_samples, step_size)
        ret[i] = calc_func(samples)
    return ret

@nb.njit(cache=True, parallel=True)
def calc2(T_arr, k, alpha, beta, n_samples, step_size=1.0):
    Vx = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    V = lambda x1,x2: Vx(x1) + Vx(-x2) + Vx(x2-x1)
    n_T = len(T_arr)
    ret = np.zeros(n_T)
    for i in nb.prange(n_T):
        samples = metropolis_samples(T_arr[i], k, alpha, beta, n_samples, step_size)
        V_arr = V(samples[:,0], samples[:,1])
        ret[i] = np.mean(V_arr)
    return ret

@nb.njit(inline='always')
def avg_x_func(samples):
    ret = np.empty(2)
    # 不能直接 axis = 0，因为 numba 不支持 axis 参数
    ret[0] = np.mean(samples[:,0])
    ret[1] = np.mean(samples[:,1])
    return ret

T_arr = np.linspace(0.0001,5,100)

k = 1.0
alpha = 2.0
beta = 1.0

n_samples = int(1e5)

# Build
_ = calc2(T_arr[:2], k, alpha, beta, n_samples, step_size=0.1)
print("Build done.")

# calc
print("Start calc.")
t_start = time.time()
E_arr = calc2(T_arr, k, alpha, beta, n_samples, step_size=0.1)
t_end = time.time()
print(f"Calc done. {t_end - t_start:.3f}")

# Plot
plt.plot(T_arr, E_arr)
plt.show()

# Build
_ = calc1(T_arr[:2], k, alpha, beta, n_samples, avg_x_func, step_size=0.1)
print("Build done.")

# calc 
print("Start calc.")
t_start = time.time()
x_avg_arr = calc1(T_arr, k, alpha, beta, n_samples, avg_x_func, step_size=0.1)
t_end = time.time()
print(f"Calc done. {t_end - t_start:.3f}")

# Plot
plt.plot(T_arr, x_avg_arr[:,0], label="0")
plt.plot(T_arr, x_avg_arr[:,1], label="1")
plt.legend()
plt.show()

