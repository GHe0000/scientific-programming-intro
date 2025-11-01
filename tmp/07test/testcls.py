import numpy as np
import numba as nb

import matplotlib.pyplot as plt

def XtoQ(X):
    return np.array([X[0]+X[1],
                     X[0]-X[1],
                     0.5*(X[2]+X[3]),
                     0.5*(X[2]-X[3])])

def QtoX(Q):
    return np.array([0.5*(Q[0]+Q[1]),
                     0.5*(Q[0]-Q[1]),
                     Q[2]+Q[3],
                     Q[2]-Q[3]])

@nb.njit(cache=True) 
def poincare(E, m, k, alpha, beta, dt, n_search, max_pt=100, Q1_section=0.0):
    dT = lambda p,m : p/m
    def force(x,k,alpha,beta):
        dV = lambda x,k,alpha,beta : k*x + alpha*x**2 + beta*x**3
        ans = np.zeros_like(x)
        x1, x2 = x[:, 0], x[:, 1]
        ans[:,0] = -dV(x1, k, alpha, beta) + dV(x2 - x1, k, alpha, beta)
        ans[:,1] = dV(-x2, k, alpha, beta) - dV(x2 - x1, k, alpha, beta)
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

    # P2_0 = 1e-5 
    # P1_0 = np.sqrt(4.0 * m * E - P2_0**2) 
    # Q = np.array([0.0, 0.0, np.sqrt(4.0*m*E), 0.0]).reshape(1, 4)
    # P20 = np.sqrt(2.0 * m * E) 
    # P10 = np.sqrt(2.0 * m * E)
    # Q = np.array([0.0, 0.0, P10, P20]).reshape(1, 4)
    Q = np.array([[0.0, 0.0, np.sqrt(2*m*E),0.0]])
    pts = np.empty((max_pt, 4))
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
                pts[npt] = Qpt
                npt += 1
            else:
                break
    return pts[:npt]

m = 1.0
k = 1.0
alpha = 2.0
beta = 1.0

E = 0.5

dt = 0.02
n_search = 500000
max_pt = 5000

res = poincare(E, m, k, alpha, beta, dt, n_search, max_pt)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt_pt = XtoQ(res)
ax.scatter(plt_pt[:,1], plt_pt[:,3], c='black', marker='.')
ax.set_xlabel('$Q_2$')
ax.set_ylabel('$P_2$')
ax.grid(True)
ax.set_title(f"E={E}")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(res[:,1], res[:,3], c='black', marker='.')
ax.set_xlabel('$Q_2$')
ax.set_ylabel('$P_2$')
ax.grid(True)
ax.set_title(f"E={E}")
plt.show()
