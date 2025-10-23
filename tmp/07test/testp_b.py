import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit(cache=True) 
def poincare_single_E(E, m, k, alpha, beta, dt, n_search, max_pt=100, Q1_section=0.0):
    
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

    P2_0 = 1e-5 
    P1_0 = np.sqrt(4.0 * m * E - P2_0**2) 
    
    # 3. 设置新的初始条件
    Q = np.array([0.0, 0.0, P1_0, P2_0]).reshape(1, 4)
    # Q = np.array([0.0, 0.0, np.sqrt(4.0*m*E), 0.0]).reshape(1, 4)
    pts = np.empty((max_pt, 2))
    npt = 0
    for _ in range(n_search):
        Q_old = Q.copy()
        #Q = Leapfrog(Q_old, m, k, alpha, beta, dt)       
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

@nb.njit(parallel=True, cache=True)
def poincare_parallel_E(E_arr, m, k, alpha, beta, dt, n_search, max_pt=500):
    n_E = len(E_arr)
    results = nb.typed.List()
    for _ in range(n_E):
        results.append(np.empty((0, 2), dtype=np.float64))
    for i in nb.prange(n_E):
        E_val = E_arr[i]
        results[i] = poincare_single_E(E_val, m, k, alpha, beta, dt, 
                                     n_search, max_pt)
        
    return results

def test():
    m = 1.0
    k = 1.0
    alpha = 0.0
    beta = 1.0
    
    dt = 0.01          # Leapfrog 是二阶，需要更小的 dt
    n_search = 200000   # 搜索步数
    max_pt = 1000       # 每个截面存储的最大点数

    n_E_steps = 500 # 扫描 100 个能量值
    E_arr = np.linspace(0.0001, 0.5, n_E_steps)
    _ = poincare_parallel_E(np.array([0.1, 0.2]), m, k, alpha, beta, dt, 1, 1)
    print("Start...")
    start_time = time.time()
    results = poincare_parallel_E(E_arr, m, k, alpha, beta, dt, 
                                n_search, max_pt)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}")
    all_E_pts = []
    all_Q2_pts = []
    all_P2_pts = []
    total_points = 0

    for i in range(len(E_arr)):
        E_val = E_arr[i]
        pts = results[i]  # (N, 2) 数组 [Q2, P2]
        
        if len(pts) > 0:
            all_E_pts.append(np.full(len(pts), E_val))
            all_Q2_pts.append(pts[:, 0])
            all_P2_pts.append(pts[:, 1])
            total_points += len(pts)

    E_flat = np.concatenate(all_E_pts)
    Q2_flat = np.concatenate(all_Q2_pts)
    P2_flat = np.concatenate(all_P2_pts)

    fig = plt.figure(figsize=(10, 7)) # 调整为 2D 图像尺寸
    ax = fig.add_subplot(111)
    ax.scatter(E_flat, Q2_flat, s=0.01, c='black', marker='.')
    ax.set_xlabel('E')
    ax.set_ylabel('$Q_2$')
    ax.grid(True)
    plt.show()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(E_flat, Q2_flat, P2_flat, s=0.1, c=E_flat, cmap='viridis', marker='.')

    ax.set_xlabel('E')
    ax.set_ylabel('$Q_2$')
    ax.set_zlabel('$P_2$')
    plt.show()


if __name__ == "__main__":
    test()
