import numpy as np
import numba as nb

@nb.njit(cache=True)
def metropolis_samples(T, k, alpha, beta, n_samples, step_size=1.0, X0=[0.0,0.0]):
    Vx = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    V = lambda x1,x2: Vx(x1) + Vx(-x2) + Vx(x1-x1)
    samples = np.zeros((2, n_samples))
    beta_T = 1 * T # kb = 1
    x1, x2 = X0
    E_old = V(x1,x2)
    for i in range(n_samples):
        x1_new = x1 + np.random.uniform(-step_size, step_size)
        x2_new = x2 + np.random.uniform(-step_size, step_size)
        E_new = V(x1_new, x2_new)
        dE = E_new - E_old
        accept = True if dE <= 0 else (np.random.random() < np.exp(-beta_T * dE))
        if accept:
            x1, x2 = x1_new, x2_new
            E_old = E_new
        samples[0,i], samples[1,i] = x1, x2
    return samples

@nb.njit(cache=True, parallel=True)
def calc_T_arr(T_arr, k, alpha, beta, n_samples, calc_func ,step_size=1.0, X0=[0.0,0.0]):
    n_T = len(T_arr)
    ret = np.zeros(n_T)
    for i in nb.prange(n_T):
        samples = metropolis_samples(T_arr[i], k, alpha, beta, n_samples, step_size, X0)
        ret[i] = calc_func(samples)
    return ret
