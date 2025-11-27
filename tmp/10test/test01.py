import numpy as np
import numba as nb

import time
import matplotlib.pyplot as plt

@nb.njit(cache=True)
def metropolis_samples_full(T, k, alpha, beta, n_samples, step_size=1.0):
    V = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    K = lambda p: 0.5 * p**2
    H = lambda X: V(X[0]) + K(X[1])
    samples = np.zeros((n_samples, 2))
    beta_T = 1.0 / (1.0*T) # kb = 1
    X = np.zeros(2)
    H_old = H(X)
    for i in range(n_samples):
        X_new = X + np.random.uniform(-step_size, step_size, size=2)
        H_new = H(X_new)
        dH = H_new - H_old
        accept = True if dH <= 0 else (np.random.random() < np.exp(-beta_T * dH))
        if accept:
            X = X_new
            H_old = H_new
        samples[i] = X
    return samples

T = 0.008
k = 1.0
alpha = 2.0
beta = 1.0

n_samples = int(1e6)

start_time = time.time()
samples = metropolis_samples_full(T, k, alpha, beta, n_samples)
print(f"{time.time() - start_time:.3f}")

sigma_x = np.sqrt(T / k)
sigma_p = np.sqrt(T)
x_pdf = lambda x: (1.0 / (sigma_x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_x)**2)
p_pdf = lambda p: (1.0 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (p / sigma_p)**2)

fig, axs = plt.subplots(1, 2, figsize=(8, 6))

axs[0].hist(samples[:, 0], color='blue', alpha=0.5, bins=100, density=True)
x_plt = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 100)
axs[0].plot(x_plt, x_pdf(x_plt), color="k")
axs[0].set_title(r'$x$')
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel(r'Density')
axs[0].grid()

axs[1].hist(samples[:, 1], color='blue', alpha=0.5, bins=100, density=True)
p_plt = np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 100)
axs[1].plot(p_plt, p_pdf(p_plt), color="k")
axs[1].set_title(r'$p$')
axs[1].set_xlabel(r'$p$')
axs[1].set_ylabel(r'Density')
axs[1].grid()

fig.suptitle(f"T={T}, k={k}, $\\alpha$={alpha}, $\\beta$={beta}, $n=10^{np.log10(n_samples):.0f}$")
plt.tight_layout()
plt.show()

def plot_heatmap(ax, x_data, y_data, bins=100):
    h, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, density=True)
    im = ax.imshow(h.T, 
                   origin='lower', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect='auto',
                   cmap='viridis')
    plt.colorbar(im, ax=ax, label="Density")
    return ax

fig, ax = plt.subplots(figsize=(8, 6))
plot_heatmap(ax, samples[:, 0], samples[:, 1])
ax.set_title(r'$x$-$p$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$p$')
plt.show()
