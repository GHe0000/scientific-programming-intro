import numpy as np
import numba as nb

import matplotlib.pyplot as plt
import matplotlib as mpl

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

T_arr = np.linspace(0.0001, 0.15, 100)
n_samples = int(1e6)
k = 1.0
alpha = 2.0
beta = 1.0

print("Build...")
_ = samples_T_arr(T_arr[:2], k, alpha, beta, n_samples)

print("Run...")
t_start = time.time()
samples_arr = samples_T_arr(T_arr, k, alpha, beta, n_samples)
print(f"Time: {time.time() - t_start:.3f}")

def plot_func(ax, samples, T_arr, n_bins=100):
    n_T, n_samples = samples.shape
    flat_data = samples.flatten()
    y_min = np.min(flat_data)
    y_max = np.max(flat_data)
    bins = np.linspace(y_min, y_max, n_bins+1)
    density = np.empty((n_T, n_bins))

    for i in range(n_T):
        hist, _ = np.histogram(samples[i,:], bins=bins, density=True)
        density[i,:] = hist
    density = density + 1e-300
    norm = mpl.colors.LogNorm(vmin=1e-10, vmax=np.max(density))
    im = ax.imshow(density.T, 
                   extent=[T_arr[0], T_arr[-1], y_min, y_max], 
                   origin='lower',
                   aspect='auto',
                   cmap='viridis',
                   norm=norm)
    plt.colorbar(im, ax=ax, label="Density(log)")
    return ax

fig, ax = plt.subplots(figsize=(8,6))
plot_func(ax, samples_arr, T_arr)
ax.set_xlabel("Temperature (T)")
ax.set_ylabel("Position (x)")
plt.show()
