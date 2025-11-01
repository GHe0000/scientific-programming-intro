import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit(cache=True)
def force(Q, k, alpha, beta):
    ans = np.zeros_like(Q)
    Q1, Q2 = Q[:, 0], Q[:, 1]
    dVdQ1 = (
        0.5 * k * Q1
        + 0.5 * alpha * Q1 * Q2
        + 0.125 * beta * Q1**3
        + 0.375 * beta * Q1 * Q2**2
    )
    dVdQ2 = (
        1.5 * k * Q2
        + 0.25 * alpha * (Q1**2 - 3 * Q2**2)
        + 0.375 * beta * (Q1**2 * Q2 + 3 * Q2**3)
    )
    
    ans[:, 0] = -dVdQ1
    ans[:, 1] = -dVdQ2
    return ans

@nb.njit(cache=True)
def Yo8_step(Q0, m, k, alpha, beta, dt):
    dT = lambda P,m : P/m
    Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
    
    C_COEFFS = np.array([0.521213104349955, 1.431316259203525, 0.988973118915378,
                         1.298883627145484, 1.216428715985135, -1.227080858951161,
                         -2.031407782603105, -1.698326184045211, -1.698326184045211,
                         -2.031407782603105, -1.227080858951161, 1.216428715985135,
                         1.298883627145484, 0.988973118915378, 1.431316259203525,
                         0.521213104349955])
    D_COEFFS = np.array([1.04242620869991, 1.82020630970714, 0.157739928123617,
                         2.44002732616735, -0.007169894197081, -2.44699182370524,
                         -1.61582374150097, -1.780828626589452, -1.61582374150097,
                         -2.44699182370524, -0.007169894197081, 2.44002732616735,
                         0.157739928123617, 1.82020630970714, 1.04242620869991])
    for i in range(15):
        P += C_COEFFS[i] * force(Q, k, alpha, beta) * dt
        Q += D_COEFFS[i] * dT(P, m) * dt
    P += C_COEFFS[15] * force(Q, k, alpha, beta) * dt
    ans = np.empty((Q0.shape[0], 4), dtype=np.float64)
    ans[:, :2], ans[:, 2:] = Q, P
    return ans

@nb.njit(cache=True)
def Leapfrog_step(Q0, m, k, alpha, beta, dt):
    dT = lambda P,m : P/m
    Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
    P += force(Q, k, alpha, beta) * (dt / 2.0)
    Q += dT(P, m) * dt
    P += force(Q, k, alpha, beta) * (dt / 2.0)
    ans = np.empty((Q0.shape[0], 4)) 
    ans[:, :2], ans[:, 2:] = Q, P
    return ans

@nb.njit(cache=True)
def lyapunov_max_single(E, m, k, alpha, beta, dt, n_sample, n_renorm, Delta0=np.array([1e-9, 0., 0., 0.]), offset=1e-300):
    P20 = np.sqrt(0.5 * m * E) 
    P10 = np.sqrt(0.5 * m * E)
    Q0 = np.array([0.0, 0.0, P10, P20])
    Q = Q0.copy()
    
    lambda_sum = 0.
    n_cycles = n_sample // n_renorm
    Qp = Q + Delta0
    d0 = np.linalg.norm(Delta0)

    Q, Qp = Q.reshape(1, 4), Qp.reshape(1, 4)
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            Q = Yo8_step(Q, m, k, alpha, beta, dt)
            Qp = Yo8_step(Qp, m, k, alpha, beta, dt)
        Delta = Qp[0] - Q[0]
        d = np.linalg.norm(Delta)
        # if d < 1e-15:
        #     d = 1e-15
        lambda_sum += np.log(np.abs(d / d0)) # 累加
        Qp = Q + (Delta / d) * d0
    total_time = n_cycles * n_renorm * dt
    return lambda_sum / total_time

@nb.njit(parallel=True, cache=True)
def lyapunov_max(E_arr, m, k, alpha, beta, dt, n_sample, n_renorm, Delta0=np.array([1e-8, 0., 0., 0.]), offset=1e-300):
    n_E = len(E_arr)
    lyapunov_arr = np.zeros(n_E, dtype=np.float64)
    for i in nb.prange(n_E):
        lyapunov_arr[i] = lyapunov_max_single(
            E_arr[i], m, k, alpha, beta, dt, n_sample, n_renorm, Delta0, offset
        )
    return lyapunov_arr

m = 1.0
k = 1.0
alpha = 0.0
beta = 0.0

nE = 50
E_arr = np.linspace(0.0001, 2, nE)

dt = 0.0001
n_sample = 200000
n_renorm = 200

Delta0 = np.array([1e-8, 0., 0., 0.])
# 预编译
_ = lyapunov_max(np.array([0.1, 0.2]), m, k, alpha, beta, dt, 2, 1)
print("Start...")
start_time = time.time()
mle_values = lyapunov_max(
    E_arr=E_arr,
    m=m,
    k=k,
    alpha=alpha,
    beta=beta,
    dt=dt,
    n_sample=n_sample,
    n_renorm=n_renorm,
    Delta0=Delta0
)

end_time = time.time()
print(f"{end_time - start_time:.2f}")
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(E_arr, mle_values, lw=2, color='b', label='$\\lambda_{max}$')
ax.axhline(0, color='red', linestyle='--', lw=1.5, label='$\\lambda_{max}=0$')
ax.set_xlim(E_arr.min(), E_arr.max())
ax.set_ylim(mle_values.min() * 0.9, mle_values.max() * 1.1)
ax.set_title(f'(m={m}, k={k}, $\\alpha$={alpha}, $\\beta$={beta})')
ax.legend()
ax.grid(True)
plt.show()
