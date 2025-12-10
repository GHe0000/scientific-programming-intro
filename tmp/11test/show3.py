t0 = time.time()

# 1. 参数设置
w, h = 800, 600
xmin, xmax = -2.0, 0.8
ymin, ymax = -1.1, 1.1

max_iter_mandel = 2000
skip_iter = 10000
keep_iter = 1000 

# 2. 计算 Mandelbrot 集合
print("Calculating Mandelbrot set...")
mask = calc_mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter_mandel)

y_indices, x_indices = np.where(mask == 0)
dx = (xmax - xmin) / w
dy = (ymax - ymin) / h
c_reals = xmin + x_indices * dx
c_imags = ymin + y_indices * dy

# 3. 采样点设置 (静态图可以设置得很高！)
# 相比 k3d 的浏览器限制，本地后台渲染可以轻松处理 20000+ 个起始点 (2000万个总点数)
max_display_points = 20000 
total_interior = len(c_reals)

print(f"Total interior candidates: {total_interior}")
print(f"Sampling {max_display_points} points for high-res static render...")

if total_interior > max_display_points:
    indices = np.random.choice(total_interior, max_display_points, replace=False)
    c_reals = c_reals[indices]
    c_imags = c_imags[indices]

xs, ys, zs = compute_orbits_flat(c_reals, c_imags, max_iter_mandel, skip_iter, keep_iter)
print(f"Orbit calc done: {time.time()-t0 :.3f}s")

# 4. 过滤与合并数据
valid_mask = (zs >= -4) & (zs <= 4)
pts = np.column_stack((xs[valid_mask], ys[valid_mask], zs[valid_mask]))

# 5. Vedo 后台渲染配置
print(f"Rendering {len(pts)} points offscreen...")

# 创建点云
# r=0.5: 点更小，看起来更细腻
# alpha=0.05: 透明度极低，通过叠加产生'烟雾'效果
actor = Points(pts, r=0.5).alpha(0.05)
actor.cmap('ocean', pts[:, 2])

# offscreen=True 是关键：不弹窗，直接在显存绘制
plt = Plotter(offscreen=True, bg='black', size=(1200, 900))

plt.add(actor)

# 设置坐标轴
axes_opts = dict(
    xtitle='Re(c)', ytitle='Im(c)', ztitle='Re(z)',
    c='white', text_scale=1.2
)

# 设置相机视角 (类似于 Matplotlib 的 view_init)
# viewup="z" 保证 Z 轴向上
# azimuth/elevation 可以通过简单的 rotate 操作模拟，或者直接用鼠标调整好的视角参数
plt.show(axes=axes_opts, viewup="z", azimuth=-60, elevation=20, zoom=1.1)

# 6. 保存并显示
save_path = 'mandelbrot_bifurcation.png'
plt.screenshot(save_path)
plt.close() # 释放资源

print(f"Image saved to {save_path}. Total time: {time.time()-t0 :.3f}s")

display(Image(filename=save_path))
