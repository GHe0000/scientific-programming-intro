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
        X = X + df(X, a, b, c) * dt

    pts = np.empty((n_search, 2))
    npt = 0
    x0 = 0.

    for _ in range(n_search):
        X_old = X.copy()
        X = X + df(X, a, b, c) * dt

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


b = 0.5
a = 0.2
c = 5.7

dt = 0.01
n_warm = 50000
n_search = 10000

start_time = time.time()
pts = rossler_poincare_single(b, a, c, dt, n_warm, n_search)
print("--- %s seconds ---" % (time.time() - start_time))

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(pts[:,0], pts[:,1], s=1)
ax.set_xlabel('y')
ax.set_ylabel('z')
plt.show()
