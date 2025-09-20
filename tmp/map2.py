import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    return r * x * (1 - x)

def plot_system_behavior(r, x0, n_iter=100):
    orbit = np.zeros(n_iter + 1)
    orbit[0] = x0
    for i in range(n_iter):
        orbit[i+1] = logistic_map(r, orbit[i])
    x_space = np.linspace(0, 1, 400)
    fx = logistic_map(r, x_space)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Logistic Map Behavior for r = {r}, x₀ = {x0}', fontsize=16)

    ax1 = axes[0]
    ax1.plot(range(n_iter + 1), orbit, 'b-o', markersize=3, alpha=0.7)
    ax1.set_title('Time Series Plot (Orbit)', fontsize=14)
    ax1.set_xlabel('Iteration (n)', fontsize=12)
    ax1.set_ylabel('State (xₙ)', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axes[1]
    ax2.plot(x_space, fx, 'r-', lw=2, label='f(x) = rx(1-x)')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='y = x')

    start_plot_index = max(0, n_iter - 100) 
    for i in range(start_plot_index, n_iter):
        # 垂直线: (x_i, x_i) -> (x_i, x_{i+1})
        ax2.plot([orbit[i], orbit[i]], [orbit[i], orbit[i+1]], 'g-', lw=0.8)
        # 水平线: (x_i, x_{i+1}) -> (x_{i+1}, x_{i+1})
        ax2.plot([orbit[i], orbit[i+1]], [orbit[i+1], orbit[i+1]], 'g-', lw=0.8)

    ax2.set_title('Phase Space & Cobweb Plot', fontsize=14)
    ax2.set_xlabel('Current State (xₙ)', fontsize=12)
    ax2.set_ylabel('Next State (xₙ₊₁)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    initial_x0 = 0.2

    plot_system_behavior(r=2.8, x0=initial_x0)

    plot_system_behavior(r=3.2, x0=initial_x0)

    plot_system_behavior(r=3.5, x0=initial_x0)

    plot_system_behavior(r=3.8, x0=initial_x0)
