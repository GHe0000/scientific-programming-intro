import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt

@nb.njit(cache=True)
def get_force(Q, k, alpha, beta):
    ans = np.zeros_like(Q)
    Q1, Q2 = Q[0], Q[1]
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
    ans[0] = -dVdQ1
    ans[1] = -dVdQ2
    return ans

@nb.njit(cache=True)
def get_K(Q_coords, k, alpha, beta):
    """
    K = dF/dQ
    [Q1, Q2]
    """
    Q1, Q2 = Q_coords[0], Q_coords[1]
    K = np.zeros((2, 2))
    
    # K[0, 0] = dF1/dQ1
    K[0, 0] = -(0.5 * k + 0.5 * alpha * Q2 + 0.375 * beta * Q1**2 + 0.375 * beta * Q2**2)
    # K[0, 1] = dF1/dQ2
    K[0, 1] = -(0.5 * alpha * Q1 + 0.75 * beta * Q1 * Q2)
    
    # K[1, 0] = dF2/dQ1 (dF2/dQ1 = dF1/dQ2)
    K[1, 0] = K[0, 1]
    # K[1, 1] = dF2/dQ2
    K[1, 1] = -(1.5 * k - 1.5 * alpha * Q2 + 0.375 * beta * Q1**2 + 3.375 * beta * Q2**2)
    
    return K

@nb.njit(cache=True)
def leapfrog_tangent_step(X, Q_mat, m, k, alpha, beta, dt):
    Q_coords = X[:2]
    P_coords = X[2:]
    
    # Phi_QQ = Q_mat[0:2, 0:2]
    # Phi_QP = Q_mat[0:2, 2:4]
    # Phi_PQ = Q_mat[2:4, 0:2]
    # Phi_PP = Q_mat[2:4, 2:4]
    Phi_QQ = Q_mat[0:2, 0:2].copy()
    Phi_QP = Q_mat[0:2, 2:4].copy()
    Phi_PQ = Q_mat[2:4, 0:2].copy()
    Phi_PP = Q_mat[2:4, 2:4].copy()
    F_old = get_force(Q_coords, k, alpha, beta)
    K_old = get_K(Q_coords, k, alpha, beta)
    P_half = P_coords + F_old * (dt / 2.0)
    Phi_PQ_half = Phi_PQ + (K_old @ Phi_QQ) * (dt / 2.0)
    Phi_PP_half = Phi_PP + (K_old @ Phi_QP) * (dt / 2.0)
    Q_new = Q_coords + (2.0 / m) * P_half * dt
    Phi_QQ_new = Phi_QQ + (2.0 / m) * Phi_PQ_half * dt
    Phi_QP_new = Phi_QP + (2.0 / m) * Phi_PP_half * dt
    F_new = get_force(Q_new, k, alpha, beta)
    K_new = get_K(Q_new, k, alpha, beta)
    P_new = P_half + F_new * (dt / 2.0)
    Phi_PQ_new = Phi_PQ_half + (K_new @ Phi_QQ_new) * (dt / 2.0)
    Phi_PP_new = Phi_PP_half + (K_new @ Phi_QP_new) * (dt / 2.0)
    X_new = np.array([Q_new[0], Q_new[1], P_new[0], P_new[1]])
    Q_mat_new = np.empty((4, 4))
    Q_mat_new[0:2, 0:2] = Phi_QQ_new
    Q_mat_new[0:2, 2:4] = Phi_QP_new
    Q_mat_new[2:4, 0:2] = Phi_PQ_new
    Q_mat_new[2:4, 2:4] = Phi_PP_new
    return X_new, Q_mat_new

# -----------------------------------------------------------------
# 李雅普诺夫指数计算
# -----------------------------------------------------------------

@nb.njit(cache=True)
def hamiltonian_lyapunov_single(E, m, k, alpha, beta, dt, n_sample, n_renorm, offset=1e-300):
    P2_0 = np.sqrt(2.0 * m * E) 
    P1_0 = np.sqrt(2.0 * m * E)
    # Q = np.array([0.0, 0.0, P1_0, P2_0]).reshape(1, 4)
    X = np.array([0.0, 0.0, P1_0, P2_0]) # 初始状态 X0

    dim = 4
    Q_mat = np.eye(dim)
    lambda_sum = np.zeros(dim)
    
    n_cycles = n_sample // n_renorm
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            X, Q_mat = leapfrog_tangent_step(X, Q_mat, m, k, alpha, beta, dt)
        Q_mat, R = np.linalg.qr(Q_mat)
        Q_mat = np.ascontiguousarray(Q_mat) # 确保内存连续，Numba需要
        diag_R = np.diag(R)
        Q_mat[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
    return lambda_sum / (n_cycles * n_renorm * dt)

@nb.njit(parallel=True, cache=True)
def hamiltonian_lyapunov_E(E_arr, m, k, alpha, beta, dt, n_sample, n_renorm, offset=1e-300):
    n_E = len(E_arr)
    lyapunov_arr = np.zeros((n_E, 4))
    for i in nb.prange(n_E):
        lyapunov_arr[i] = hamiltonian_lyapunov_single(E_arr[i], m, k, alpha, beta, 
                                                       dt, n_sample, n_renorm, offset)
    return lyapunov_arr

if __name__ == "__main__":
    
    m = 1.0
    k = 1.0
    alpha = 0.0
    beta = 1.0
    
    nE = 500
    E_arr = np.linspace(0.0001, 0.5, nE)

    dt = 0.01
    n_sample = 100000
    n_renorm = 10

    start_time = time.time()
    lyapunov_arr = hamiltonian_lyapunov_E(E_arr, m, k, alpha, beta, 
                                        dt, n_sample, n_renorm)
    print(f"{time.time() - start_time}")

    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(4):
        ax.plot(E_arr, lyapunov_arr[:,i], lw=1, alpha=0.7, label=f'$\\lambda_{i+1}$')
    ax.axhline(0, color='red', linestyle='--', lw=1.5, label='$\\lambda=0$')
    ax.set_xlim(E_arr.min(), E_arr.max())
    ax.set_title(f'm={m}, k={k}, alpha={alpha}, beta={beta}')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.show()
