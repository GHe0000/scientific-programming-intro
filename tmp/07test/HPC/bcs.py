
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit(cache=True) 
def poincare_single(E, m, k, alpha, beta, dt, n_search, max_pt=100, Q1_section=0.0):
    dT = lambda P, m: 2.0 * P/m
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

    def Yo8_step(Q0,m,k,alpha,beta,dt):
        Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
        # Yoshida H. Construction of higher order symplectic integrators[J]. Physics letters A, 1990, 150(5-7): 262-268.
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
            P += C_COEFFS[i] * force(Q,k,alpha,beta) * dt
            Q += D_COEFFS[i] * dT(P,m) * dt
        P += C_COEFFS[15] * force(Q,k,alpha,beta) * dt
        ans = np.empty((Q0.shape[0], 4))
        ans[:,:2], ans[:,2:] = Q, P
        return ans
    
    def Leapfrog(Q0, m, k, alpha, beta, dt):
        Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
        P += force(Q, k, alpha, beta) * (dt / 2.0)
        Q += dT(P, m) * dt
        P += force(Q, k, alpha, beta) * (dt / 2.0)
        ans = np.empty((Q0.shape[0], 4)) 
        ans[:, :2], ans[:, 2:] = Q, P
        return ans

    P20 = np.sqrt(0.5 * m * E) 
    P10 = np.sqrt(0.5 * m * E)
    Q = np.array([0.0, 0.0, P10, P20]).reshape(1, 4)
    pts = np.empty((max_pt, 2))
    npt = 0
    for _ in range(n_search):
        Q_old = Q.copy()
        # Q = Leapfrog(Q_old, m, k, alpha, beta, dt)       
        Q = Yo8_step(Q_old, m, k, alpha, beta, dt)
        Q1_new = Q[0][0]
        Q1_old = Q_old[0][0]
        P1_old = Q_old[0][2]
        if Q1_old < Q1_section and Q1_new >= Q1_section and P1_old > 0.0:
            s = (Q1_section - Q1_old) / (Q1_new - Q1_old)
            if npt < max_pt:
                Qpt = Q_old + s * (Q - Q_old)
                pts[npt, 0] = Qpt[0][1]
                pts[npt, 1] = Qpt[0][3]
                npt += 1
            else:
                break
    return pts[:npt]

m = 1.0
k = 1.0
alpha = 2.0
beta = 1.0

dt = 0.005
n_search = 500000
max_pt = 5000

E = 0.74

start_time = time.time()
results = poincare_single(E, m, k, alpha, beta, dt, n_search, max_pt)
print(f"Time taken: {time.time() - start_time} seconds")

plt.scatter(results[:, 0], results[:, 1], s=0.1)
plt.show()
