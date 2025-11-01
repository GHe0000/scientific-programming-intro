import numpy as np
import numba as nb


@nb.njit(cache=True)
def lyapunov_single(E, m, k, alpha, beta, dt, n_sample, n_renorm, offset=1e-300):
    dT = lambda P, m: 2.0 * P/m

    def genMP(P, m, k, alpha, beta):
        MP = np.empty((4,4))
        return MP

    def genMQ(Q, m, k, alpha, beta):
        MQ = np.empty((4,4))
        return MQ

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
    
    def leapfrog(X0, Q0, m, k, alpha, beta, dt):
        # X0->Xans
        Q, P = X0[:2].copy(), X0[2:].copy()
        P += force(Q, k, alpha, beta) * (dt / 2.0)
        Q += dT(P, m) * dt
        P += force(Q, k, alpha, beta) * (dt / 2.0)
        Xans = np.empty(X0.shape) 
        Xans[:2], Xans[2:] = Q, P
        # Q0->Qans
        I1 = Q0[0:2, 0:2]
        I2 = Q0[2:4, 2:4]


        Qans = np.zeros(Q0.shape)
        Qans[0:2, 0:2] = I1
        Qans[2:4, 2:4] = I2
        return Xans, Qans


    P20 = np.sqrt(2.0 * m * E) 
    P10 = np.sqrt(2.0 * m * E)
    X = np.array([0.0, 0.0, P10, P20])

    dim = len(X)
    Q = np.eye(dim)
    lambda_sum = np.zeros(dim)
    n_cycles = n_sample // n_renorm
    for cycle in range(n_cycles):
        for _ in range(n_renorm):
            X, Q = leapfrog(X, Q, m, k, alpha, beta, dt)
        Q, R = np.linalg.qr(Q)
        Q = np.ascontiguousarray(Q) # 确保内存连续，Numba需要
        diag_R = np.diag(R)
        Q[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
    return lambda_sum / (n_cycles * n_renorm * dt)
