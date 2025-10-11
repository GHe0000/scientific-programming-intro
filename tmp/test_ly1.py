import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# 使用您提供的函数，只需确保已经导入了 numba
@nb.njit()
def lyapunov_nd(df_func, jac_func, x0, dt, n_warm, n_step, n_renorm, offset=1e-300):
    """
    计算N维系统的李雅普诺夫指数。
    """
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
    # 如果 n_cycle 为 0，则直接返回，避免后续错误
    if n_cycle == 0:
        return np.array([0.0]), np.zeros((1, dim))
        
    t_save = np.arange(1, n_cycle + 1) * n_renorm * dt
    lambda_save = np.zeros((n_cycle, dim))

    for cycle in range(n_cycle):
        for _ in range(n_renorm):
            # 状态向量 x 的演化
            k1 = df_func(x)
            k2 = df_func(x + 0.5 * dt * k1)
            k3 = df_func(x + 0.5 * dt * k2)
            k4 = df_func(x + dt * k3)
            x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 扰动向量 Q 的演化
            J1 = jac_func(x) @ Q
            J2 = jac_func(x + 0.5 * dt * k1) @ (Q + 0.5 * dt * J1)
            J3 = jac_func(x + 0.5 * dt * k2) @ (Q + 0.5 * dt * J2)
            J4 = jac_func(x + dt * k3) @ (Q + dt * J3)
            Q += (dt / 6.0) * (J1 + 2 * J2 + 2 * J3 + J4)

        # Gram-Schmidt 正交化
        Q, R = np.linalg.qr(Q)
        diag_R = np.diag(R)
        
        # 确保 Q 矩阵的定向一致性
        Q[:, diag_R < 0] *= -1
        
        lambda_sum += np.log(np.abs(diag_R) + offset) # 防止 log(0)
        lambda_save[cycle] = lambda_sum / t_save[cycle]
        
    return t_save, lambda_save

# --- 洛伦兹系统定义 ---
SIGMA = 10.0
BETA = 8.0 / 3.0

@nb.njit()
def lorenz_df(x, r):
    """洛伦兹系统的微分方程"""
    dx_dt = SIGMA * (x[1] - x[0])
    dy_dt = x[0] * (r - x[2]) - x[1]
    dz_dt = x[0] * x[1] - BETA * x[2]
    return np.array([dx_dt, dy_dt, dz_dt])

@nb.njit()
def lorenz_jac(x, r):
    """洛伦兹系统的雅可比矩阵"""
    return np.array([
        [-SIGMA, SIGMA, 0],
        [r - x[2], -1, -x[0]],
        [x[1], x[0], -BETA]
    ])

# --- 主计算函数 ---
def calculate_lyapunov_for_r_arr(r_arr, x0, dt, n_warm, n_step, n_renorm):
    """
    为一系列r值计算李雅普诺夫指数。
    返回: (n, 3) 形状的数组，其中 n 是 r_arr 的长度。
    """
    lyap_exponents = np.zeros((len(r_arr), 3))
    
    for i, r in enumerate(r_arr):
        print(f"正在计算 r = {r:.2f}...")
        
        # 使用闭包来固定参数 r
        @nb.njit
        def df_func(x):
            return lorenz_df(x, r)

        @nb.njit
        def jac_func(x):
            return lorenz_jac(x, r)
        
        # 运行计算
        t, lambdas = lyapunov_nd(df_func, jac_func, x0, dt, n_warm, n_step, n_renorm)
        
        # 我们取最后计算出的稳定值作为最终的李雅普诺夫指数
        if len(lambdas) > 0:
            lyap_exponents[i, :] = lambdas[-1, :]
        
    return lyap_exponents

# --- 参数设置和执行 ---
if __name__ == '__main__':
    # 计算参数
    dt = 0.01
    n_warm = 1000      # 预热步数，让轨道进入吸引子
    n_step = 50000     # 总计算步数
    n_renorm = 10      # 重新标准化的频率

    # 初始条件
    x0 = np.array([1.0, 1.0, 1.0])
    
    # r值的范围
    r_arr = np.arange(1.0, 40.0, 0.5)
    
    # 执行计算
    lyapunov_spectrum = calculate_lyapunov_for_r_arr(r_arr, x0, dt, n_warm, n_step, n_renorm)
    
    # --- 绘图 ---
    plt.figure(figsize=(12, 7))
    
    # 绘制三个李雅普诺夫指数
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    for i in range(3):
        plt.plot(r_arr, lyapunov_spectrum[:, i], lw=2, label=labels[i])
        
    # 添加一条 y=0 的参考线
    plt.axhline(0, color='k', linestyle='--', lw=1)
    
    # 图像美化
    plt.title('Lyapunov Exponents of Lorenz System vs. Parameter r', fontsize=16)
    plt.xlabel('Parameter r', fontsize=12)
    plt.ylabel('Lyapunov Exponents ($\lambda$)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.ylim(-15, 5) # 设置y轴范围以便更好地观察
    
    # 保存图像
    plt.savefig("lyapunov_spectrum_vs_r.png")
    
    print("\n计算完成，图像已保存为 'lyapunov_spectrum_vs_r.png'")
