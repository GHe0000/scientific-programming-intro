import numpy as np 

def plot_cobweb(ax, map_func, x_range, x_n, x0, n):
    '''
    绘制迭代方程 map_func 的蛛网图

    参数：
    ax: maplotlib.axes
        需要绘制的轴对象
    map_func: callable
        迭代函数，只有一个传入参数
    x_range: (x_min, x_max)
        需要绘制的范围
    x_n: int 或 float
        采样点数量
    x0: float
        迭代初始值
    n: int 或 float
        迭代次数

    返回：
    ax: maplotlib.axes
        绘制的轴对象
    '''
    n = int(n)
    x_arr = np.linspace(x_range[0], x_range[1], int(x_n))
    path = np.zeros((2*n+1, 2))
    path[0] = np.array([x0, x0])
    xp = x0
    for i in range(n):
        path[2*i+1] = np.array([xp, map_func(xp)])
        xp = map_func(xp)
        path[2*i+2] = np.array([xp, xp])
    ax.plot(path[:,0], path[:,1], '--o', markersize=4, label="迭代过程")
    ax.plot(x_arr, x_arr, label="$x_{n+1}=x_n$")
    ax.plot(x_arr, map_func(x_arr), label="$x_{n+1}=f(x_n)$")
    return ax
