import numpy as np
from scipy.optimize import brentq

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

def find_superstable_r_coarse_fine(period, search_start, search_end=1.402, step=1e-6):
    """
    使用“粗筛 + 细化”的方法寻找超稳定环的 r 值。
    
    Args:
        period (int): 环的周期。
        search_start (float): 粗筛的起始 r 值。
        search_end (float): 粗筛的结束 r 值。
        step (float): 粗筛的步长。
        
    Returns:
        float or None: 求解得到的 r 值，如果找不到则返回 None。
    """
    equation_to_solve = lambda r: iterate_map(0, r, period)
    
    # --- 1. 粗筛阶段 ---
    print(f"  [粗筛] 周期 {period}: 从 r={search_start:.6f} 开始搜索...")
    r_current = search_start
    y_current = equation_to_solve(r_current)
    
    bracket = None
    while r_current < search_end:
        r_next = r_current + step
        y_next = equation_to_solve(r_next)
        
        # 检查符号是否改变，如果改变则找到了包含根的区间
        if y_current * y_next < 0:
            bracket = [r_current, r_next]
            print(f"  [粗筛] 成功! 在 [{r_current:.6f}, {r_next:.6f}] 中发现根。")
            break
        
        r_current = r_next
        y_current = y_next
        
    if bracket is None:
        print(f"  [粗筛] 失败! 未能在 r < {search_end} 的范围内找到根。")
        return None
        
    # --- 2. 细化阶段 ---
    try:
        # 使用 brentq 在粗筛找到的区间内进行精确求解
        root = brentq(equation_to_solve, bracket[0], bracket[1])
        return root
    except ValueError:
        print(f"  [细化] 失败! brentq 未能在区间 {bracket} 内收敛。")
        return None

def main():
    R_values = []

    # 已知的解析解
    r0 = 0.0
    R_values.append(r0)
    r1 = 1.0
    R_values.append(r1)

    # 从 n=2 (周期 4) 开始进行数值计算
    max_n = 10 
    
    for n in range(2, max_n + 1):
        period = 2**n
        
        # 搜索的起点是上一个找到的 R_{n-1} 值，加上一个很小的偏移量以避免重复
        search_start_r = R_values[-1] + 1e-9
        
        # 调用“粗筛+细化”函数
        r_n = find_superstable_r_coarse_fine(period, search_start_r)
        
        if r_n is not None:
            R_values.append(r_n)
        else:
            print(f"计算在周期 {period} 处中断。")
            break
            
    # 打印最终结果
    print("\n" + "="*70)
    print("最终计算结果")
    print("="*70)
    print(f"{'n':<5} {'Period (2^n)':<15} {'Parameter r (R_n)':<25} {'Delta Estimate':<20}")
    print("-" * 70)

    for i in range(len(R_values)):
        period = 2**i
        if i < 2:
            print(f"{i:<5} {period:<15} {R_values[i]:<25.15f} {'N/A'}")
        else:
            delta_estimate = (R_values[i-1] - R_values[i-2]) / (R_values[i] - R_values[i-1])
            print(f"{i:<5} {period:<15} {R_values[i]:<25.15f} {delta_estimate:<20.15f}")
    
    print("-" * 70)
    print(f"费根鲍姆常数 (公认值): δ ≈ 4.66920160910299067185")


if __name__ == "__main__":
    main()
