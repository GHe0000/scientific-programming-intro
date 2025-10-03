import numpy as np # 数值计算库
import numba as nb # 引入 jit 来加速函数（如果需要）
import sympy as sym # 符号计算库
import matplotlib.pyplot as plt # 图像绘制
from matplotlib.backends.backend_agg import FigureCanvasAgg # 设置图像后端用
import ipywidgets as ipw # 交互控件
from IPython.display import display, Math, Latex # 打印数学公式

# 设置随机数种子确保结果的可复现性
np.random.seed(3407)

# 使用 mathjax 来在 jupyter notebook 显示数学公式
sym.init_printing(use_latex='mathjax')

# 设置 matplotlib 绘制的图像嵌入到 jupyter notebook 的方式
%matplotlib widget

# 一个工具函数，可以让静态图片不经过 widget 直接嵌入 Jupyter notebook
# 这样图片可以直接存到 nb 文件里，而不是需要运行才能显示（类似 inline）
def display_inline(fig):
    fig.set_canvas(FigureCanvasAgg(fig))
    display(fig)
    plt.close(fig) # 释放 fig，减小资源消耗

# 一个工具函数，用于在 Jupyter 中快速实现公式和字符串混合输出
def display_math(*args):
    parts = []
    for a in args:
        if isinstance(a, str):
            parts.append(a)
        else:
            parts.append(sym.latex(a))
    display(Math("".join(parts)))

# 设置 matplotlib 使用的字体，避免出现中文问题
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# 设置默认坐标轴字体大小
plt.rcParams['axes.labelsize'] = 14
