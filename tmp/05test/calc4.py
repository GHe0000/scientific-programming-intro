import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit()
def rossler_lyapunov_max_single(b, a, c, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), Delta0=np.array([1e-8,0.,0.]), offset=1e-300):
    def df(X,a,b,c):
        return np.array([-X[1]-X[2],
                     X[0]+a*X[1],
                     b+X[2]*(X[0]-c)])
    X = X0.copy()
    for _ in range(n_warm):
        X = X + df(X, a, b, c) * dt
        # k1 = df(X, a, b, c)
        # k2 = df(X + 0.5 * dt * k1, a, b, c)
        # k3 = df(X + 0.5 * dt * k2, a, b, c)
        # k4 = df(X + dt * k3, a, b, c)
        # X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    lambda_sum = 0.
    n_cycles = n_sample // n_renorm
    X_p = X + Delta0
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            X = X + df(X, a, b, c) * dt
            X_p = X_p + df(X_p, a, b, c) * dt
            # k1 = df(X, a, b, c)
            # k2 = df(X + 0.5 * dt * k1, a, b, c)
            # k3 = df(X + 0.5 * dt * k2, a, b, c)
            # k4 = df(X + dt * k3, a, b, c)
            # X += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # k1 = df(X_p, a, b, c)
            # k2 = df(X_p + 0.5 * dt * k1, a, b, c)
            # k3 = df(X_p + 0.5 * dt * k2, a, b, c)
            # k4 = df(X_p + dt * k3, a, b, c)
            # X_p += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        Delta = X - X_p
        d0 = np.linalg.norm(Delta0)
        d = np.linalg.norm(Delta)
        lambda_sum += np.log(np.abs(d/d0) + offset) # 防止 log(0)
        X_p = X + (Delta / d) * np.linalg.norm(Delta0)
    return lambda_sum / (n_cycles * n_renorm * dt)

@nb.njit(parallel=True, cache=True)
def rossler_lyapunov_max(b_arr, a, c, dt, n_warm, n_sample, n_renorm, X0=np.array([0.,0.,0.]), Delta0=np.array([1e-8,0.,0.]), offset=1e-300):
    n_r = len(b_arr)
    lyapunov_arr = np.zeros(n_r, dtype=np.float64)
    for i in nb.prange(n_r):
        lyapunov_arr[i] = rossler_lyapunov_max_single(b_arr[i], a, c, dt, n_warm, n_sample, n_renorm, X0, Delta0, offset)
    return lyapunov_arr
a = 0.2
c = 5.7
b_arr = np.linspace(0.2, 2.0, 400)

dt = 0.01          # 积分时间步长
n_warm = 50000     # 预热步数 (演化 200 个时间单位)
n_sample = 500000  # 采样总步数 (演化 5000 个时间单位)
n_renorm = 10      # 重正化间隔 (每 0.1 个时间单位)

start_time = time.time()

mle_values = rossler_lyapunov_max(
    b_arr=b_arr, 
    a=a, 
    c=c, 
    dt=dt, 
    n_warm=n_warm, 
    n_sample=n_sample, 
    n_renorm=n_renorm
)

end_time = time.time()
print(f"计算完成，耗时: {end_time - start_time:.2f} 秒")
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(b_arr, mle_values, lw=2, color='b', label='$\\lambda_{max}$')
ax.axhline(0, color='red', linestyle='--', lw=1.5, label='$\\lambda_{max}=0$')
ax.set_xlim(b_arr.min(), b_arr.max())
ax.legend()
plt.show()
