import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t, u, sigma, rho, beta):
    x, y, z = u
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

initial_state = [0.0, 0.05, 0]

t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(
    fun=lorenz,
    t_span=t_span,
    y0=initial_state,
    args=(sigma, rho, beta),
    dense_output=True,
    t_eval=t_eval
)

x, y, z = solution.y

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, lw=0.5, color='blue')
ax.set_title(f"$\\sigma={sigma:.4f}$, $\\rho={rho:.4f}$, $\\beta={beta:.4f}$")
plt.show()
