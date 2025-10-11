'''
Lyapunov 指数相关
'''

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

@nb.njit()
def lyapunov_nd(df_func, jac_func, x0, dt, n_warm, n_step, n_renorm, offset=1e-300):

    x = x0.copy()
    # 预热
    for _ in range(n_warm):
        k1 = df_func(x)
        k2 = df_func(x + 0.5 * dt * k1)
        k3 = df_func(x + 0.5 * dt * k2)
        k4 = df_func(x + dt * k3)       
        x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    dim = x0.size
    Q = np.eye(dim)

    lambda_sum = np.zeros(dim)
    n_cycle = n_step // n_renorm
    t_save = np.arange(1, n_cycle+1) * n_renorm * dt
    lambda_save = np.zeros((n_cycle, dim))

    for cycle in range(n_cycle):
        for _ in range(n_renorm):
            k1 = df_func(x)
            k2 = df_func(x + 0.5 * dt * k1)
            k3 = df_func(x + 0.5 * dt * k2)
            k4 = df_func(x + dt * k3)

            J1 = jac_func(x) @ Q
            J2 = jac_func(x + 0.5 * dt * k1) @ (Q + 0.5 * dt * J1)
            J3 = jac_func(x + 0.5 * dt * k2) @ (Q + 0.5 * dt * J2)
            J4 = jac_func(x + dt * k3) @ (Q + dt * J3)
            
            x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            Q += (dt / 6.0) * (J1 + 2 * J2 + 2 * J3 + J4)
        
        # 正交化
        Q, R = np.linalg.qr(Q)
        Q = np.ascontiguousarray(Q) # 转为 C 顺序，从而加速
        diag_R = np.diag(R)
        Q[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
        lambda_save[cycle] = lambda_sum / t_save[cycle]
    return t_save, lambda_save

sigma, rho, beta = 10.0, 28.0, 8.0/3.0

@nb.njit()
def lorenz_rhs(x):
    dx = np.empty(3)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0]*(rho - x[2]) - x[1]
    dx[2] = x[0]*x[1] - beta*x[2]
    return dx

@nb.njit()
def lorenz_jac(x):
    J = np.empty((3,3))
    J[0,0] = -sigma;   J[0,1] = sigma; J[0,2] = 0.0
    J[1,0] = rho-x[2]; J[1,1] = -1.0;  J[1,2] = -x[0]
    J[2,0] = x[1];     J[2,1] = x[0];  J[2,2] = -beta
    return J

# 稳定点

def gen_stable_pt(sigma, rho, beta):
    X1 = np.array([0,0,0])

x0 = np.array([1.0, 1.0, 1.0])
dt = 0.01
n_step = 10000
renorm_step = 10

t, lams = lyapunov_nd(lorenz_rhs,
                      lorenz_jac,
                      x0, dt, 10000, n_step, renorm_step)
print(lams[-1])

# 绘图
plt.figure(figsize=(8,5))
for i in range(lams.shape[1]):
    plt.plot(t, lams[:,i], label=f"λ_{i+1}")
plt.axhline(0, color="k", linestyle="--")
plt.legend()
plt.show()

