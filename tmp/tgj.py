import mpmath as mp

mp.mp.dps = 80  # 全局精度，可按需提高

def forward_iter_and_derivs(x, r, k):
    """
    计算 p = f^k(x), 以及以下导数（关于初值 x 和参数 r）：
      A = d(f^k)/dx
      B = d(f^k)/dr
      C = d^2(f^k)/dx^2
      D = d(d(f^k)/dx)/dr = d^2(f^k)/(dr dx)
    返回 (p, A, B, C, D)
    递推基准： p0 = x, A0 = 1, B0 = 0, C0 = 0, D0 = 0
    映射 f(x,r) = r - x^2
    f_x = -2*x, f_xx = -2, f_r = 1
    """
    p = mp.mpf(x)
    A = mp.mpf(1)   # dp/dx
    B = mp.mpf(0)   # dp/dr
    C = mp.mpf(0)   # d2p/dx2
    D = mp.mpf(0)   # d(d p / dx)/dr

    for _ in range(k):
        fx = -2 * p         # f_x evaluated at current p
        fxx = mp.mpf(-2)    # f_xx is constant -2
        # update p
        p_new = r - p*p
        # update derivatives using chain rule for composition
        # A_{n+1} = f_x(p) * A
        A_new = fx * A
        # B_{n+1} = f_x(p) * B + f_r  (f_r = 1)
        B_new = fx * B + mp.mpf(1)
        # C_{n+1} = f_xx * A^2 + f_x * C
        C_new = fxx * (A*A) + fx * C
        # D_{n+1} = f_xx * A * B + f_x * D   (since f_xr = 0)
        D_new = fxx * (A * B) + fx * D

        p, A, B, C, D = p_new, A_new, B_new, C_new, D_new

    return p, A, B, C, D

def newton_bifurcation(n, x0, r0, tol=1e-30, maxiter=50, prec=80, verbose=False):
    """
    二维牛顿法求解 (F1=0, F2=0)：
      F1(x,r) = f^{2^n}(x,r) - x
      F2(x,r) = d/dx f^{2^n}(x,r) + 1
    参数:
      n      : 计算 f^{2^n}
      x0,r0  : 初始猜测 (mpf 或可转换为 mpf)
      tol    : 终止容限（相对/绝对）
      maxiter: 最大牛顿迭代次数
      prec   : mpmath 精度（小数位）
    返回 (x, r, converged, info)
    """
    mp.mp.dps = prec
    x = mp.mpf(x0)
    r = mp.mpf(r0)
    k = 2**n

    for it in range(1, maxiter+1):
        p, A, B, C, D = forward_iter_and_derivs(x, r, k)

        F1 = p - x         # f^k(x) - x
        F2 = A + 1         # d/dx f^k(x) + 1

        # Jacobian J = [[dF1/dx, dF1/dr],
        #               [dF2/dx, dF2/dr]]
        J11 = A - 1        # dF1/dx = A - 1
        J12 = B            # dF1/dr = B
        J21 = C            # dF2/dx = C
        J22 = D            # dF2/dr = D

        # Build jacobian matrix and rhs
        J = mp.matrix([[J11, J12], [J21, J22]])
        F = mp.matrix([F1, F2])

        # Solve J * delta = -F
        # check singularity / conditioning
        try:
            delta = mp.lu_solve(J, -F)
        except Exception as e:
            return (x, r, False, f"Jacobian solve failed at iter {it}: {e}")

        dx = delta[0]
        dr = delta[1]

        x += dx
        r += dr

        if verbose:
            print(f"iter {it}: x={mp.nstr(x,20)}, r={mp.nstr(r,20)}, |dx|={mp.nstr(abs(dx),5)}, |dr|={mp.nstr(abs(dr),5)}")

        # 收敛判断（同时看变量和残差）
        if abs(dx) < tol and abs(dr) < tol and abs(F1) < tol and abs(F2) < tol:
            return (x, r, True, f"converged in {it} iters")

    return (x, r, False, f"no convergence in {maxiter} iters; last residuals F1={mp.nstr(F1,10)}, F2={mp.nstr(F2,10)}")

if __name__ == "__main__":
    mp.mp.dps = 80
    # 例：求第 1 次倍周期分叉（2^1 周期，即 period-2 出现）：已知理论 r1 = 3/4 = 0.75，固定点 x≈0.5
    x0 = mp.mpf('0.5')
    r0 = mp.mpf('0.75')
    x, r, ok, info = newton_bifurcation(n=1, x0=x0, r0=r0, tol=mp.mpf('1e-30'), maxiter=50, prec=80, verbose=True)
    print("result:", ok, info)
    print("x =", mp.nstr(x,30))
    print("r =", mp.nstr(r,30))

    # 例：求第 2 次倍周期分叉（2^2 周期，r2理论约 = 1.25）
    x0 = mp.mpf('0.9')   # 给定一个大致的周期-4 点初值（经验）
    r0 = mp.mpf('1.25')  # 近似 r2
    x, r, ok, info = newton_bifurcation(n=2, x0=x0, r0=r0, tol=mp.mpf('1e-30'), maxiter=80, prec=80, verbose=True)
    print("result:", ok, info)
    print("x =", mp.nstr(x,30))
    print("r =", mp.nstr(r,30))
