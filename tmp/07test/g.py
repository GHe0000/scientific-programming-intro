# 使用 Yoshida8 求解
# 后面会广泛使用，因此进行了向量化优化，可以直接传入数组
@nb.njit(cache=True)
def Yo8(X0,t_eval,m,k,alpha,beta,dt):
    # X0 = [x1,x2,p1,p2]
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
    dT = lambda p,m : p/m
    def force(x,k,alpha,beta):
        dV = lambda x,k,alpha,beta : k*x + alpha*x**2 + beta*x**3
        ans = np.zeros_like(x)
        x1, x2 = x[:, 0], x[:, 1]
        ans[:,0] = -dV(x1, k, alpha, beta) + dV(x2 - x1, k, alpha, beta)
        ans[:,1] = dV(-x2, k, alpha, beta) - dV(x2 - x1, k, alpha, beta)
        return ans
    x, p = X0[:, :2].copy(), X0[:, 2:].copy()
    sol = np.zeros((len(t_eval), X0.shape[0], 4))
    sol[0,:,:2], sol[0,:,2:] = x,p
    for tn in range(1,len(t_eval)):
        for i in range(15):
            p += C_COEFFS[i] * force(x,k,alpha,beta) * dt
            x += D_COEFFS[i] * dT(p,m) * dt
        p += C_COEFFS[15] * force(x,k,alpha,beta) * dt
        sol[tn,:,:2],sol[tn,:,2:] = x,p
    return sol

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
    for i in range(15):
        P += C_COEFFS[i] * force(Q,k,alpha,beta) * dt
        Q += D_COEFFS[i] * dT(P,m) * dt
    P += C_COEFFS[15] * force(Q,k,alpha,beta) * dt
    ans = np.empty((Q0.shape[0], 4))
    ans[:,:2], ans[:,2:] = Q, P
    return ans
