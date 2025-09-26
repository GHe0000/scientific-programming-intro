'''
Symplectic Partitioned Runge-Kutta
8 阶的辛分区 Runge-Kutta 法求解器

使用 Yoshida 给出的高精度 8 阶辛积分器系数
'''
import numpy as np
import numba as nb

def _Yo8_core(gradT, gradV, q0, p0, dt, n_step):
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
    q_save = np.zeros((n_step + 1, len(q0)))
    p_save = np.zeros((n_step + 1, len(p0)))

    q_save[0] = q0
    p_save[0] = p0

    q = q0.copy()
    p = p0.copy()

    for i in range(n_step):
        for j in range(15):
            p -= C_COEFFS[i] * gradV(q) * dt
            q += D_COEFFS[i] * gradT(p) * dt
        p -= C_COEFFS[15] * gradV(q) * dt
        q_save[i+1] = q
        p_save[i+1] = p
    return q_save, p_save

def Yo8(
        gradT: callable[[np.ndarray], np.ndarray],
        gradV: callable[[np.ndarray], np.ndarray],
        q0: np.ndarray,
        p0: np.ndarray,
        t: float,
        dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一个通用的 8 阶辛分区 Runge-Kutta 法 Hamilton 方程求解器.

    此函数使用由 Laskar & Robutel 提出的 8 阶辛积分方法，对形如
    H(q,p) = T(p) + V(q) 的可分离哈密顿系统进行数值积分.
    
    与原始版本不同，此函数接受动能关于动量的梯度 (dT/dp) 和
    势能关于位置的梯度 (dV/dq) 作为输入，使其更具通用性.

    函数会自动检查梯度函数是否被 Numba 编译，以实现高性能计算.

    Parameters
    ----------
    gradT : callable
        计算动能梯度 dT/dp 的函数.
        函数签名为 `f(p) -> np.ndarray`.
        为了获得最佳性能，此函数应由 Numba 的 njit 装饰器编译.
        
    gradV : callable
        计算势能梯度 dV/dq 的函数.
        函数签名为 `f(q) -> np.ndarray`.
        为了获得最佳性能，此函数应由 Numba 的 njit 装饰器编译.
        
    q0 : np.ndarray
        初始位置.
        
    p0 : np.ndarray
        初始动量.
        
    t : float
        总计算时间.
        
    dt : float
        每个积分步长的时间间隔.

    Returns
    -------
    t_eval : np.ndarray
        从 0 到总积分时间的时刻数组，形状为 `(n_step + 1,)`.
    q : np.ndarray
        位置的轨迹数组，形状为 `(n_step + 1, N)`.
    p : np.ndarray
        动量的轨迹数组，形状为 `(n_step + 1, N)`.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    n_step = int(t / dt)
    t_eval = np.linspace(0, t, n_step + 1)

    # 检查两个函数是否都被 Numba 编译
    is_jitted = (isinstance(gradT, nb.core.dispatcher.Dispatcher) and
                 isinstance(gradV, nb.core.dispatcher.Dispatcher))

    if is_jitted:
        loop_func = nb.njit(_Yo8_core)
    else:
        print("Warning: One or both gradient functions are not numba-compiled, may be slow.")
        loop_func = _SPRK_core
    q, p = loop_func(gradT, gradV, q0, p0, dt, n_step)
    return t_eval, q, p

@nb.njit()
def Yo8_step(q, p, m, dt):
    q = q.copy()
    p = p.copy()

    def gradT(p, m):
        return p / m

    def gradV(q):
        dV = lambda x: x + alpha * x**2
        q_m1 = np.zeros(N)
        q_m1[1:] = q[:-1]
        q_p1 = np.zeros(N)
        q_p1[:-1] = q[1:]
        return dV(q - q_m1) - dV(q_p1 - q)

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
        p -= C_COEFFS[i] * gradV(q) * dt
        q += D_COEFFS[i] * gradT(p, m) * dt
    p -= C_COEFFS[15] * gradV(q) * dt
    return q, p
