import numpy as np
import numba as nb

@nb.njit()
def lorentz_lyapunov_single(r, sigma, beta, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), offset=1e-300):
    def df(X, r, sigma, beta):
        return np.array([sigma * (X[1] - X[0]),
                         X[0] * (r - X[2]) - X[1],
                         X[0] * X[1] - beta * X[2]])

    def jac(X, r, sigma, beta):
        return np.array([[-sigma, sigma, 0    ],
                         [r-X[2], -1,    -X[0]],
                         [X[1]  , X[0], -beta ]])

    X = X0.copy()
    for _ in range(n_warm):
        k1 = df(X, r, sigma, beta)
        k2 = df(X + 0.5 * dt * k1, r, sigma, beta)
        k3 = df(X + 0.5 * dt * k2, r, sigma, beta)
        k4 = df(X + dt * k3, r, sigma, beta)
        X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dim = len(X)
    Q = np.eye(dim)
    lambda_sum = np.zeros(dim)
    n_cycles = n_sample // n_renorm
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            k1 = df(X, r, sigma, beta)
            k2 = df(X + 0.5 * dt * k1, r, sigma, beta)
            k3 = df(X + 0.5 * dt * k2, r, sigma, beta)
            k4 = df(X + dt * k3, r, sigma, beta)
            X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            J1 = jac(X, r, sigma, beta) @ Q
            J2 = jac(X + 0.5 * dt * k1, r, sigma, beta) @ (Q + 0.5 * dt * J1)
            J3 = jac(X + 0.5 * dt * k2, r, sigma, beta) @ (Q + 0.5 * dt * J2)
            J4 = jac(X + dt * k3, r, sigma, beta) @ (Q + dt * J3)
            Q += (dt / 6.0) * (J1 + 2 * J2 + 2 * J3 + J4)
        Q, R = np.linalg.qr(Q)
        Q = np.ascontiguousarray(Q)
        diag_R = np.diag(R)
        Q[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
    return lambda_sum / (n_cycles * n_renorm * dt)

@nb.njit(parallel=True, cache=True)
def lorentz_lyapunov_nr(r_arr, sigma, beta, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), offset=1e-300):
    n_r = len(r_arr)
    lyapunov_arr = np.zeros((n_r, len(X0)))
    for i in nb.prange(n_r):
        r_val = r_arr[i]
        lyapunov_arr[i] = lorentz_lyapunov_single(r_val, sigma, beta, dt, n_warm, n_sample, n_renorm, X0, offset)
    return lyapunov_arr

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

X0 = np.array([0.0, 0.05, 0])

dt = 0.01
n_warm = 10000
n_search = 10000

r_arr = np.linspace(0,32,1000)

lyapunov_arr = lorentz_lyapunov_nr(r_arr, sigma, beta, dt, n_warm, n_search, 10, X0)
import matplotlib.pyplot as plt
plt.plot(r_arr, lyapunov_arr)
plt.show()
