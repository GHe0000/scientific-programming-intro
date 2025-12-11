import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit(cache=True)
def lorenz_sample(skip_n, save_n, per_n, dt=0.01):
    def lorenz(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])
        
    def rk4_step(state, dt):
        k1 = lorenz(state)
        k2 = lorenz(state + k1 * dt * 0.5)
        k3 = lorenz(state + k2 * dt * 0.5)
        k4 = lorenz(state + k3 * dt)
        return state + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
        
    X = np.array([0.1, 0.0, 0.0])
    
    # 预热阶段：迭代但不保存，为了让系统进入吸引子轨道
    for _ in range(skip_n):
        X = rk4_step(X, dt)
        
    ret = np.empty((save_n, 3), dtype=np.float64)
    
    # 采样阶段
    for i in range(save_n):
        ret[i] = X
        # 在两次保存之间迭代 per_n 次
        for _ in range(per_n):
            X = rk4_step(X, dt)
            
    return ret

skip_n = 5000    # 跳过前 5000 步
save_n = 10000   # 保存 10000 个点
per_n = 5        # 每计算 5 步保存 1 个点 (相当于降采样)
dt = 0.01

traj = lorenz_sample(skip_n, save_n, per_n, dt)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], 
                s=0.5, c=traj[:, 2], cmap='viridis', alpha=0.8)
ax.set_title(f"Lorenz Attractor 3D Distribution (N={save_n})")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.colorbar(sc, label='Z Value', pad=0.1)
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()
