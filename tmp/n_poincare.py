import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import time

@nb.njit()
def lorentz_poincare_single(r, sigma, beta, dt, n_warm, n_search, max_pt=100, X0=np.array([0.,0.,0.])):
    
    def df(X, r, sigma, beta):
        return np.array([sigma * (X[1] - X[0]),
                         X[0] * (r - X[2]) - X[1],
                         X[0] * X[1] - beta * X[2]])

    # 预热
    X = X0.copy()
    for _ in range(n_warm):
        k1 = df(X, r, sigma, beta)
        k2 = df(X + 0.5 * dt * k1, r, sigma, beta)
        k3 = df(X + 0.5 * dt * k2, r, sigma, beta)
        k4 = df(X + dt * k3, r, sigma, beta)
        X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    z0 = r - 1.0 # 截面
    pts = np.empty((n_search, 2))
    npt = 0

    for _ in range(n_search):
        X_old = X.copy()
        k1 = df(X, r, sigma, beta)
        k2 = df(X + 0.5 * dt * k1, r, sigma, beta)
        k3 = df(X + 0.5 * dt * k2, r, sigma, beta)
        k4 = df(X + dt * k3, r, sigma, beta)
        X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        z_old = X_old[2]
        z_new = X[2]

        if z_old < z0 and z_new >= z0:
            s = (z0 - z_old) / (z_new - z_old)
            if npt < max_pt:
                pts[npt] = (X_old + s * (X - X_old))[:2]
                npt += 1
            else:
                break
    return pts[:npt]

@nb.njit(parallel=True, cache=True)
def lorentz_poincare_nr(r_arr, sigma, beta, dt, n_warm, n_search, max_pt=500, X0=np.array([1.,1.,1.])):
    n_r = len(r_arr)
    results = nb.typed.List()
    for _ in range(n_r):
        results.append(np.empty((0, 2), dtype=np.float64))
    for i in nb.prange(n_r):
        r_val = r_arr[i]
        results[i] = lorentz_poincare_single(r_val, sigma, beta, dt, n_warm, n_search, max_pt, X0)
    return results

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

X0 = np.array([0.0, 0.05, 0])

dt = 0.01
n_warm = 10000
n_search = 10000

r_arr = np.linspace(0,100,10000)

start_time = time.time()
res = lorentz_poincare_nr(r_arr, sigma, beta, dt, n_warm, n_search, X0=X0)
print("--- %s seconds ---" % (time.time() - start_time))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# for i in range(len(r_arr)):
#     r_plot = np.ones_like(res[i][:,0]) * r_arr[i]
#     ax.scatter(r_plot, res[i][:,0], res[i][:,1], c='k', s=0.1, alpha=0.2)
# ax.set_box_aspect((3,1,1))
# ax.view_init(30, -70)
# ax.set_xlabel('r')
# ax.set_ylabel('x')
# ax.set_zlabel('y')
# plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(r_arr)):
    r_plot = np.ones_like(res[i][:,0]) * r_arr[i]
    ax.scatter(r_plot, res[i][:,1], c='k', s=0.1, alpha=0.2)
ax.set_xlabel('r')
ax.set_ylabel('y')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(r_arr)):
    r_plot = np.ones_like(res[i][:,0]) * r_arr[i]
    ax.scatter(r_plot, res[i][:,0], c='k', s=0.1, alpha=0.2)
ax.set_xlabel('r')
ax.set_ylabel('x')
plt.show()
