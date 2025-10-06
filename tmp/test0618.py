import numpy as np
import numba as nb

# 使用 Numba 的 @njit 装饰器进行即时编译，并开启并行化
@nb.njit(parallel=True)
def golden_section_search_parallel(bounds, precision=1e-8, max_iter=5000):
    def objective_function(r):
        x = 0.0
        for _ in range(5000):
            x = r - x**2
        
        sum_ln = 0.0
        for _ in range(2000):
            x = r - x**2
            sum_ln += np.log(2 * np.abs(x) + 1e-300)
        return np.abs(sum_ln / 2000)

    GR = (np.sqrt(5) - 1) / 2
    n_params = bounds.shape[1]
    
    # 初始化结果数组，用于存放每个区间的最优解
    results = np.zeros(n_params, dtype=np.float64)

    # 使用 numba.prange 进行并行循环
    for i in nb.prange(n_params):
        a = bounds[0, i]
        b = bounds[1, i]

        c = b - GR * (b - a)
        d = a + GR * (b - a)
        
        f_c = objective_function(c)
        f_d = objective_function(d)

        # 3. 开始迭代收缩区间
        for _ in range(max_iter):
            if (b - a) < precision:
                break
            if f_c < f_d: # 注意这里是不等号方向，取决于您是找最大值还是最小值
                b = d
                d = c
                c = b - GR * (b - a)
                f_d = f_c
                f_c = objective_function(c)
            else:
                a = c
                c = d
                d = a + GR * (b - a)
                f_c = f_d
                f_d = objective_function(d)
                
        results[i] = (a + b) / 2
        
    return results
bounds_to_test = np.array([[0.74, 0.76],[1.24,1.26]]).T 
ans = golden_section_search_parallel(bounds_to_test)
exact = np.array([0.75, 1.25])
print(ans)
print(ans-exact)
