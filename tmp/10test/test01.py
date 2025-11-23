import numpy as np
import numba as nb

import time
import matplotlib.pyplot as plt

@nb.njit(cache=True)
def metropolis_samples_full(T, k, alpha, beta, n_samples, step_size=1.0):
    Vx = lambda x:0.5 * k * x**2 + (1.0/3.0) * alpha * x**3 + 0.25 * beta * x**4
    V = lambda x1,x2: Vx(x1) + Vx(-x2) + Vx(x2-x1)
    K = lambda p1,p2: 0.5 * p1**2 + 0.5 * p2**2
    H = lambda X: V(X[0], X[1]) + K(X[2], X[3])
    samples = np.zeros((n_samples, 4))
    beta_T = 1.0 / (1.0*T) # kb = 1
    X = np.zeros(4)
    H_old = H(X)
    for i in range(n_samples):
        X_new = X + np.random.uniform(-step_size, step_size, size=4)
        H_new = H(X_new)
        dH = H_new - H_old
        accept = True if dH <= 0 else (np.random.random() < np.exp(-beta_T * dH))
        if accept:
            X = X_new
            H_old = H_new
        samples[i] = X
    return samples

T = 2.0
k = 1.0
alpha = 0.0
beta = 0.0

n_samples = int(1e6)

start_time = time.time()
samples = metropolis_samples_full(T, k, alpha, beta, n_samples)
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f}s")

p_pdf = lambda p: (1 / np.sqrt(2 * np.pi * T)) * np.exp(-0.5 * p**2 / (1.0 * T))
x_pdf = lambda x: (1.0 / np.sqrt(2 * np.pi * (2*T/3))) * np.exp(-x**2 / (2 * (2*T/3)))
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

axs[0, 0].hist(samples[:, 0], color='blue', alpha=0.5, bins=100, density=True)
# x_plt = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 2]), 100)
# axs[0, 0].plot(x_plt, x_pdf(x_plt), color="k")
axs[0, 0].set_title(r'$x_1$')

axs[0, 1].hist(samples[:, 1], color='blue', alpha=0.5, bins=100, density=True)
# x_plt = np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 100)
# axs[0, 1].plot(x_plt, x_pdf(x_plt), color="k")
axs[0, 1].set_title(r'$x_2$')

axs[1, 0].hist(samples[:, 2], color='blue', alpha=0.5, bins=100, density=True)
x_plt = np.linspace(np.min(samples[:, 2]), np.max(samples[:, 2]), 100)
axs[1, 0].plot(x_plt, p_pdf(x_plt), color="k")
axs[1, 0].set_title(r'$p_1$')

axs[1, 1].hist(samples[:, 3], color='blue', alpha=0.5, bins=100, density=True)
x_plt = np.linspace(np.min(samples[:, 3]), np.max(samples[:, 3]), 100)
axs[1, 1].plot(x_plt, p_pdf(x_plt), color="k")
axs[1, 1].set_title(r'$p_2$')

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

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_heatmap(axs[0], samples[:, 0], samples[:, 1])
axs[0].set_title(r'$x_1$ vs $x_2$')
axs[0].set_xlabel(r'$x_1$')
axs[0].set_ylabel(r'$x_2$')

plot_heatmap(axs[1], samples[:, 2], samples[:, 3])
axs[1].set_title(r'$p_1$ vs $p_2$')
axs[1].set_xlabel(r'$p_1$')
axs[1].set_ylabel(r'$p_2$')

fig.suptitle(f"T={T}, k={k}, $\\alpha$={alpha}, $\\beta$={beta}, $n=10^{np.log10(n_samples):.0f}$")
plt.tight_layout()
plt.show()
