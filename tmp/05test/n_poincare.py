import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import time

@nb.njit()
def rossler_poincare_single(b, a, c, dt, n_warm, n_search, max_pt=100, X0=np.array([0.,0.,0.])):
    def df(X,a,b,c):
        return np.array([-X[1]-X[2],
                     X[0]+a*X[1],
                     b+X[2]*(X[0]-c)])
    X = X0.copy()
    for _ in range(n_warm):
        k1 = df(X, a, b, c)
        k2 = df(X + 0.5 * dt * k1, a, b, c)
        k3 = df(X + 0.5 * dt * k2, a, b, c)
        k4 = df(X + dt * k3, a, b, c)
        X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    pts = np.empty((n_search, 2))
    npt = 0
    x0 = 0.

    for _ in range(n_search):
        X_old = X.copy()
        k1 = df(X, a, b, c)
        k2 = df(X + 0.5 * dt * k1, a, b, c)
        k3 = df(X + 0.5 * dt * k2, a, b, c)
        k4 = df(X + dt * k3, a, b, c)
        X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_old = X_old[0]
        x_new = X[0]
        if x_old < x0 and x_new >= x0:
            s = (x0 - x_old) / (x_new - x_old)
            if npt < max_pt:
                pts[npt] = (X_old + s * (X - X_old))[1:]
                npt += 1
            else:
                break
    return pts[:npt]

@nb.njit(parallel=True, cache=True)
def rossler_poincare_nr(b_arr, a, c, dt, n_warm, n_search, max_pt=500, X0=np.array([1.,1.,1.])):
    n_r = len(b_arr)
    results = nb.typed.List() # 告诉 JIT 这里结果的类型
    for _ in range(n_r):
        results.append(np.empty((0, 2), dtype=np.float64))
    for i in nb.prange(n_r):
        b_val = b_arr[i]
        results[i] = rossler_poincare_single(b_val, a, c, dt, n_warm, n_search, max_pt, X0)
    return results



b_arr = np.linspace(0.2, 2.0, 1000)
a = 0.2
c = 5.7

dt = 0.01
n_warm = 10000
n_search = 20000

max_pt = 5000

X0 = np.array([2.5,0.,0.])

start_time = time.time()
res = rossler_poincare_nr(b_arr, a, c, dt, n_warm, n_search, max_pt=max_pt, X0=X0)
print("--- %s seconds ---" % (time.time() - start_time))


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(b_arr)):
    b_plot = np.ones_like(res[i][:,0]) * b_arr[i]
    ax.scatter(b_plot, res[i][:,0], res[i][:,1], c='k', s=0.1, alpha=0.2)
ax.set_box_aspect((3,1,1))
ax.view_init(30, -70)
ax.set_xlabel('r')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#
# fig, ax = plt.subplots(figsize=(8, 6))
# for i in range(len(r_arr)):
#     r_plot = np.ones_like(res[i][:,0]) * r_arr[i]
#     ax.scatter(r_plot, res[i][:,1], c='k', s=0.1, alpha=0.2)
# ax.set_xlabel('r')
# ax.set_ylabel('y')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8, 6))
# for i in range(len(r_arr)):
#     r_plot = np.ones_like(res[i][:,0]) * r_arr[i]
#     ax.scatter(r_plot, res[i][:,0], c='k', s=0.1, alpha=0.2)
# ax.set_xlabel('r')
# ax.set_ylabel('x')
# plt.show()
