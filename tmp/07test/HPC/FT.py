import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

@nb.njit(cache=True)
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

@nb.njit(cache=True)
def Yo8_step(Q0, m, k, alpha, beta, dt):
    Q, P = Q0[:, :2].copy(), Q0[:, 2:].copy()
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
        P += C_COEFFS[i] * force(Q, k, alpha, beta) * dt
        Q += D_COEFFS[i] * (2.0 * P/m) * dt
    P += C_COEFFS[15] * force(Q, k, alpha, beta) * dt
    ans = np.empty((Q0.shape[0], 4), dtype=np.float64)
    ans[:, :2], ans[:, 2:] = Q, P
    return ans

@nb.njit(cache=True)
def calc_single(E, m, k, alpha, beta, dt_sim, dt_sample, n_sample):
    n_steps_per_sample = int(round(dt_sample / dt_sim))
    if n_steps_per_sample < 1:
        n_steps_per_sample = 1
    P10 = np.sqrt(0.5 * m * E)
    P20 = np.sqrt(0.5 * m * E)
    Q = np.array([[0.0, 0.0, P10, P20]], dtype=np.float64)
    ans = np.zeros(n_sample, dtype=np.float64)
    ans[0] = Q[0, 0] + Q[0, 1]
    for i in range(1, n_sample):
        for _ in range(n_steps_per_sample):
            Q = Yo8_step(Q, m, k, alpha, beta, dt_sim)
        ans[i] = Q[0, 0] + Q[0, 1] # 记录 q1 + q2
    return ans

@nb.njit(parallel=True, cache=True)
def calc_n(E_arr, m, k, alpha, beta, dt_sim, dt_sample, n_sample):
    nE = len(E_arr)
    ans = np.zeros((n_sample, nE), dtype=np.float64)
    for i in nb.prange(nE):
        ans[:, i] = calc_single(
            E_arr[i], m, k, alpha, beta, dt_sim, dt_sample, n_sample
        )
    return ans

m = 1.0
k = 1.0
alpha = 2.0
beta = 1.0

nE = 2000
E_arr = np.linspace(0.0001, 2, nE)
#E_arr = np.linspace(0.65, 0.75, nE)

dt_sample = 0.02
dt_sim = 0.0005
n_sample = 2**18

n_steps_per_sample = int(round(dt_sample / dt_sim))
print(f"Simulation step (dt_sim):     {dt_sim}")
print(f"Sampling step (dt_sample):    {dt_sample}")
print(f"FFT sample points (n_sample): {n_sample}")
print(f"Steps per sample:             {n_steps_per_sample}")
print(f"Total simulation time (T):    {n_sample * dt_sample:.2f} s")
print(f"FFT Freq. Resolution (1/T):   {1.0 / (n_sample * dt_sample):.2e} Hz")
print(f"FFT Nyquist Freq. (1/2*dt_s): {1.0 / (2 * dt_sample):.2f} Hz")

print("\nCompiling Numba functions...")
_ = calc_n(E_arr[:2], m, k, alpha, beta, dt_sim, dt_sample, 10)
print("Compilation finished.")
print(f"Start parallel computation for {nE} energies...")
start_time = time.time()

q_sum_traj = calc_n(E_arr, m, k, alpha, beta, dt_sim, dt_sample, n_sample)

end_time = time.time()
print(f"Computation finished in {end_time - start_time:.2f} seconds.")

print("Calculating FFT...")
window = np.hanning(n_sample)[:, np.newaxis]
q_sum_traj_windowed = q_sum_traj * window
spectra = np.fft.rfft(q_sum_traj_windowed, axis=0)
amplitudes = np.log10(np.abs(spectra) / np.mean(window) + 1e-300)
freqs = np.fft.rfftfreq(n_sample, d=dt_sample) 

f_min = 0.0
f_max = 0.5

freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
freqs_to_plot = freqs[freq_indices]
amplitudes_to_plot = amplitudes[freq_indices, :]

fig, ax = plt.subplots(figsize=(12, 7))
c = ax.pcolormesh(
    E_arr, 
    freqs_to_plot,
    amplitudes_to_plot,
    shading='gouraud', 
    cmap='viridis'
)
fig.colorbar(c, ax=ax, label='Amplitude(log10)')

ax.set_xlabel('$E$')
ax.set_ylabel('$f$ (Hz)')
ax.set_title(f'Spectrogram (m={m}, k={k}, $\\alpha$={alpha}, $\\beta$={beta})')
ax.grid(True, linestyle='--', alpha=0.5)
ax.axvline(0.7153, linestyle='--', c='r', lw=2., label="0.7153")
ax.axvline(1.0098, linestyle='--', c='r', lw=2., label="1.0098")
ax.legend(loc='upper left')
plt.savefig("FTa2_2.png", dpi=300, bbox_inches='tight')
plt.show()
