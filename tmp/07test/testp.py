import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import time 

@nb.njit(cache=True)
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
    dT = lambda P,m : P/(2.0 * m)
    def force(Q, k, alpha, beta):
        ans = np.zeros_like(Q)
        Q1, Q2 = Q[:, 0], Q[:, 1]
        beta_term = Q1**2 + 3*Q2**2
        dV_dQ1 = 2*k*Q1 + 4*alpha*Q1*Q2 + 2*beta*Q1*beta_term
        dV_dQ2 = 6*k*Q2 + 2*alpha*(Q1**2 - 3*Q2**2) + 6*beta*Q2*beta_term
        ans[:, 0] = -dV_dQ1
        ans[:, 1] = -dV_dQ2
        return ans
    for i in range(15):
        P += C_COEFFS[i] * force(Q,k,alpha,beta) * dt
        Q += D_COEFFS[i] * dT(P,m) * dt
    P += C_COEFFS[15] * force(Q,k,alpha,beta) * dt
    ans = np.empty((Q0.shape[0], 4))
    ans[:,:2], ans[:,2:] = Q, P
    return ans

@nb.njit()
def poincare(m, k, alpha, beta, dt, n_warm, n_search, max_pt=100, Q0=np.array([0.,0.,1.,0.])):
    Q = Q0.copy().reshape(1,4)
    for _ in range(n_warm):
        Q = Yo8_step(Q,m,k,alpha,beta,dt)
    pts = np.empty((max_pt, 2))
    npt = 0
    for _ in range(n_search):
        Q_old = Q.copy()
        Q = Yo8_step(Q_old,m,k,alpha,beta,dt)
        xQ, xQ_old = Q[0][0], Q_old[0][0]
        if xQ_old < 0 and xQ >= 0:
            s = -xQ_old / (xQ - xQ_old)
            if npt < max_pt:
                Qpt = Q_old + s * (Q - Q_old)
                pts[npt,0] = Qpt[0][1]
                pts[npt,1] = Qpt[0][3]
                npt += 1
            else:
                break
    return pts[:npt]

def run_single_test():
    m = 1.0
    k = 1.0
    alpha = 1.0  # 非线性参数
    beta = 1.0   # 非线性参数
    
    dt = 0.01
    n_warm = 5000       # 预热步数
    n_search = 200000  # 搜索步数
    max_pt = 5000      # 最大存储点数

    Q0_init = np.array([0.0, 0.0, 2.0, 0.0])
    
    E_total = Q0_init[2]**2 / (4*m) + Q0_init[3]**2 / (4*m) 
    
    print(f"开始计算: 非线性系统 (alpha={alpha}, beta={beta}, E={E_total})...")
    start_time = time.time()
    pts = poincare(m, k, alpha, beta, dt, 
                   n_warm, n_search, max_pt, Q0_init)
    print(f"计算完成, 耗时: {time.time() - start_time:.2f}s, 找到 {len(pts)} 个点")

    plt.figure(figsize=(8, 8)) # 单个图形，调整大小
    
    if len(pts) > 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=0.5, c='red', marker='.')
    else:
        print("警告：未找到任何截面点。")

    plt.title(f'Poincaré Section ($Q_1 = 0, P_1 > 0$) \n $E={E_total}, \\alpha={alpha}, \\beta={beta}$')
    plt.xlabel('$Q_2$')
    plt.ylabel('$P_2$')
    plt.grid(True)
    plt.axis('equal') # 保持纵横比
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_single_test()
