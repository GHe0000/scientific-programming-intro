import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit(cache=True)
def Yo8Q(Q0,t_eval,m,k,alpha,beta,dt):
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
    Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
    sol = np.zeros((len(t_eval), Q0.shape[0], 4))
    sol[0,:,:2], sol[0,:,2:] = Q,P
    for tn in range(1,len(t_eval)):
        for i in range(15):
            P += C_COEFFS[i] * force(Q,k,alpha,beta) * dt
            Q += D_COEFFS[i] * dT(P,m) * dt
        P += C_COEFFS[15] * force(Q,k,alpha,beta) * dt
        sol[tn,:,:2],sol[tn,:,2:] = Q,P
    return sol

m = 1.0
k = 1.0
alpha = 0.0
beta = 0.0

# E = 0.477825
E = 0.5

dt = 0.001
n_sample = 2**16

t_eval = np.arange(n_sample) * dt
Q0 = np.zeros((1, 4), dtype=np.float64)
P10 = np.sqrt(0.5 * m * E)
P20 = np.sqrt(0.5 * m * E)

Q0[:, 2] = P10
Q0[:, 3] = P20

_ = Yo8Q(Q0[:2, :], t_eval[:2], m, k, alpha, beta, dt)

print("Start")
start_time = time.time()
sol = Yo8Q(Q0, t_eval, m, k, alpha, beta, dt)
end_time = time.time()
print(f"{end_time - start_time:.2f}")

# q1_trajectories = sol[:, :, 0]
# spectra = np.fft.rfft(q1_trajectories, axis=0)

q1_traj = sol[:, 0, 0]
q2_traj = sol[:, 0, 1]
q_sum_traj = q1_traj + q2_traj # 两个轨迹相加
window = np.hanning(n_sample)
q_sum_traj_windowed = q_sum_traj * window

spectra = np.fft.rfft(q_sum_traj_windowed, axis=0) # 对总和进行 FFT
# amplitudes = np.abs(spectra) / np.mean(window)
amplitudes = np.log10(np.abs(spectra) / np.mean(window) + 1e-300)
freqs = np.fft.rfftfreq(n_sample, d=dt)

f_min = 0.0
f_max = 0.5
freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]

freqs_to_plot = freqs[freq_indices]
amplitudes_to_plot = amplitudes[freq_indices] # 直接使用 1D 的 amplitudes

# --- 开始 1D 绘图 ---
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(freqs_to_plot, amplitudes_to_plot, lw=1.5, color='blue')

ax.set_xlim(f_min, f_max)
ymin = np.min(amplitudes_to_plot) - (np.max(amplitudes_to_plot) - np.min(amplitudes_to_plot)) * 0.1
ymax = np.max(amplitudes_to_plot) * 1.1
ax.set_ylim(ymin, ymax)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 打印占比最大的前 5 个频率
max_indices = np.argsort(amplitudes_to_plot)[::-1][:5]
print(f"Top 5 frequencies: {freqs_to_plot[max_indices]}")

# 计算前两个频率的比值
ratio = amplitudes_to_plot[max_indices[0]] / amplitudes_to_plot[max_indices[1]]
print(f"{ratio}")
