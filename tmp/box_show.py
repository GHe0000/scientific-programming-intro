import numpy as np
import matplotlib.pyplot as plt

# 准备螺旋线数据
t = np.linspace(0, 5 * np.pi, 200)
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 10, 200)

fig = plt.figure(figsize=(15, 5))

# --- 视角1: 默认视角 ---
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x, y, z)
ax1.set_title('Default View')
ax1.set_box_aspect((3, 1, 1)) # 使用上一问的长方体设置

# --- 视角2: 自定义视角 (elev=45, azim=45) ---
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(x, y, z)
ax2.view_init(elev=30, azim=-70) # 设置俯仰角和方位角
ax2.set_box_aspect((3, 1, 1))

# --- 视角3: 鸟瞰图 (Top-down view) ---
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(x, y, z)
ax3.view_init(elev=90, azim=0) # 从正上方看
ax3.set_title('Top-down View (elev=90)')
# 鸟瞰图通常用正方形的盒子比较好
ax3.set_box_aspect((3, 1, 1)) 
print(f"Best view found: elev={ax1.elev}, azim={ax1.azim}")

plt.tight_layout()
plt.show()

# 专业建议：如何找到最佳视角参数？
# 1. 先用鼠标交互式地旋转，直到找到你最满意的角度。
# 2. 在代码的 plt.show() 之前，打印出当前的 elev 和 azim 值。
# print(f"Best view found: elev={ax.elev}, azim={ax.azim}")
# 3. 将这两个值复制到你的 ax.view_init() 函数中，这样你就可以永久保存这个视角了！
