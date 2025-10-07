import numpy as np

# 核心的映射函数和迭代函数保持不变
def logistic_map(x, r):
    """
    定义迭代映射: x_{n+1} = r - x_{n}^2
    """
    return r - x**2

def iterate_map(x0, r, n):
    """
    从 x0 开始，对给定的 r 值进行 n 次迭代。
    """
    x = x0
    for _ in range(n):
        x = logistic_map(x, r)
    return x

def bisection_refine(func, bracket, tol=1e-15, max_iter=100):
    """
    使用自定义的二分查找（Bisection Method）在给定区间内寻找根。
    """
    a, b = bracket[0], bracket[1]
    y_a = func(a)

    if y_a * func(b) > 0:
        return None

    for _ in range(max_iter):
        if (b - a) < tol:
            return a + (b - a) / 2
        midpoint = a + (b - a) / 2
        y_mid = func(midpoint)
        if y_a * y_mid < 0:
            b = midpoint
        else:
            a = midpoint
            y_a = y_mid
    return a + (b - a) / 2

def find_superstable_r_coarse_fine(period, search_start, search_end=1.402, step=1e-6):
    """
    使用“粗筛 + 细化”的方法寻找超稳定环的 r 值。
    """
    equation_to_solve = lambda r: iterate_map(0, r, period)
    
    # --- 1. 粗筛阶段 ---
    r_current = search_start
    y_current = equation_to_solve(r_current)
    
    bracket = None
    while r_current < search_end:
        r_next = r_current + step
        if r_next > search_end: r_next = search_end
        y_next = equation_to_solve(r_next)
        if y_current * y_next < 0:
            bracket = [r_current, r_next]
            break
        r_current = r_next
        y_current = y_next
        if r_current == search_end: break
        
    if bracket is None:
        return None
        
    # --- 2. 细化阶段 ---
    return bisection_refine(equation_to_solve, bracket)

# --- 第 1 步: 计算超稳定环参数 R_n (与之前完全相同) ---
R_values = []
R_values.extend([0.0, 1.0]) # R_0 和 R_1

max_n = 12
print("开始计算超稳定环参数 R_n...")
for n in range(2, max_n + 1):
    period = 2**n
    search_start_r = R_values[-1] + 1e-9
    r_n = find_superstable_r_coarse_fine(period, search_start_r)
    if r_n is not None:
        R_values.append(r_n)
        print(f"  周期 {period:<5} -> R_{n} = {r_n:.15f}")
    else:
        print(f"计算在周期 {period} 处中断。")
        break

# --- 第 2 步: 使用 R_n 计算距离 d_n ---
d_values = []
# 注意：d_n 的定义从 n=1 (周期 2) 开始
# d_1, d_2, d_3, ...
for n in range(1, len(R_values)):
    period_for_d = 2**(n - 1)
    r_val = R_values[n]
    d_n = iterate_map(0, r_val, period_for_d)
    d_values.append(d_n)

# --- 第 3 步: 打印包含两个常数估算的最终结果 ---
print("\n" + "="*90)
print("最终计算结果")
print("="*90)
print(f"{'n':<3} {'Period':<8} {'Parameter r (R_n)':<24} {'Distance (d_n)':<22} {'Delta Est.':<15} {'Alpha Est.':<15}")
print("-" * 90)

for i in range(len(R_values)):
    period = 2**i
    r_val_str = f"{R_values[i]:.15f}"
    
    # d_n 从 n=1 开始，所以 d_values 的索引比 R_values 的索引小 1
    d_val_str = "N/A"
    if i > 0:
        d_val_str = f"{d_values[i-1]:.15f}"

    # delta 估计从 n=2 开始
    delta_str = "N/A"
    if i >= 2:
        delta_est = (R_values[i-1] - R_values[i-2]) / (R_values[i] - R_values[i-1])
        delta_str = f"{delta_est:.8f}"
    
    # alpha 估计从 n=2 开始 (需要 d_1 和 d_2)
    alpha_str = "N/A"
    if i >= 2:
        # alpha_i = d_{i-1} / d_i
        alpha_est = d_values[i-2] / d_values[i-1]
        alpha_str = f"{alpha_est:.8f}"

    print(f"{i:<3} {period:<8} {r_val_str:<24} {d_val_str:<22} {delta_str:<15} {alpha_str:<15}")

print("-" * 90)
print(f"费根鲍姆常数 (公认值):")
print(f"  δ ≈ 4.6692016...")
print(f"  α ≈ -2.5029078...")
