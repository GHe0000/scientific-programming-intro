import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt

@nb.njit()
def rossler_lyapunov_single(b, a, c, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), offset=1e-300):
    def df(X,a,b,c):
        return np.array([-X[1]-X[2],
                     X[0]+a*X[1],
                     b+X[2]*(X[0]-c)])

    def jac(X,a,b,c):
        return np.array([[0., -1.,      -1.],
                         [1.,   a,       0.],
                         [X[2], 0., -c+X[0]]])
    X = X0.copy()
    for _ in range(n_warm):
        X = X + df(X, a, b, c) * dt
        # k1 = df(X, a, b, c)
        # k2 = df(X + 0.5 * dt * k1, a, b, c)
        # k3 = df(X + 0.5 * dt * k2, a, b, c)
        # k4 = df(X + dt * k3, a, b, c)
        # X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dim = len(X)
    Q = np.eye(dim)
    lambda_sum = np.zeros(dim)
    n_cycles = n_sample // n_renorm
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            X_old = X.copy()
            X = X + df(X_old, a, b, c) * dt
            Q = Q + dt * jac(X_old, a, b, c) @ Q
            # k1 = df(X, a, b, c)
            # k2 = df(X + 0.5 * dt * k1, a, b, c)
            # k3 = df(X + 0.5 * dt * k2, a, b, c)
            # k4 = df(X + dt * k3, a, b, c)
            # X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # J1 = jac(X, a, b, c) @ Q
            # J2 = jac(X + 0.5 * dt * k1, a, b, c) @ (Q + 0.5 * dt * J1)
            # J3 = jac(X + 0.5 * dt * k2, a, b, c) @ (Q + 0.5 * dt * J2)
            # J4 = jac(X + dt * k3, a, b, c) @ (Q + dt * J3)
            # Q += (dt / 6.0) * (J1 + 2 * J2 + 2 * J3 + J4)
        Q, R = np.linalg.qr(Q)
        Q = np.ascontiguousarray(Q)
        diag_R = np.diag(R)
        Q[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
    return lambda_sum / (n_cycles * n_renorm * dt)

@nb.njit(parallel=True, cache=True)
def rossler_lyapunov_nr(b_arr, a, c, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), offset=1e-300):
    n_r = len(b_arr)
    lyapunov_arr = np.zeros((n_r, len(X0)))
    for i in nb.prange(n_r):
        lyapunov_arr[i] = rossler_lyapunov_single(b_arr[i], a, c, dt, n_warm, n_sample, n_renorm, X0, offset)
    return lyapunov_arr

a = 0.2
c = 5.7
b_arr = np.linspace(0.2,2,400)
X0 = np.array([0.0, 0.05, 0])

dt = 0.01
n_warm = 20000
n_search = 200000

start_time = time.time()
lyapunov_arr = rossler_lyapunov_nr(b_arr, a, c, dt, n_warm, n_search, 10, X0)
print("--- %s seconds ---" % (time.time() - start_time))

lyapunov_max_arr = np.max(lyapunov_arr, axis=1)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(b_arr, lyapunov_max_arr, lw=2, color='b', label='$\\lambda_{max}$')
ax.axhline(0, color='red', linestyle='--', lw=1.5, label='$\\lambda_{max}=0$')
ax.set_xlim(b_arr.min(), b_arr.max())
ax.legend()
plt.show()
