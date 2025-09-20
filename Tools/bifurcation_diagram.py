import numpy as np

def bifurcation_diagram(ax, map_func, r_range, x_range, n_step, n_sample, **kwargs):
    r = np.linspace(r_range[0], r_range[1], r_range[2])
    x = np.random.uniform(x_range[0], x_range[1], x_range[2])
    r_tmp = r.reshape(r_range[2], 1) # 转为列向量，为了利用 broadcast

    # 到达稳定状态
    for _ in range(n_step):
        x = map_func(r_tmp, x)

    result = np.zeros((r_range[2], x_range[2], n_sample))
    for i in range(n_sample):
        x = map_func(r_tmp, x)
        result[:, :, i] = x
    x_plot = result.reshape(r_range[2], -1)
    ax.plot(r, x_plot, **kwargs)
    ax.set_ylim(x_range)
    ax.set_xlim(r_range)
    return ax, r, x_plot
