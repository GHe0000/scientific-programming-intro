import numpy as np
import numba as nb

@nb.njit(parallel=True)
def refine_by_0618(r_range_arr, precision=1e-8, max_iter=5000):
    def lyapunov(r):
        x = 0.0
        for _ in range(10000):
            x = r - x**2
        sum_ln = 0.0
        for _ in range(10000):
            x = r - x**2
            sum_ln += np.log(2 * np.abs(x))
        return sum_ln / 10000

    phi = (np.sqrt(5) - 1) / 2
    n_r = r_range_arr.shape[0]
    results = np.zeros(n_r, dtype=np.float64)
    for i in nb.prange(n_r):
        a = r_range_arr[i, 0]
        b = r_range_arr[i, 1]
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        f_c = lyapunov(c)
        f_d = lyapunov(d)

        for _ in range(max_iter):
            if (b - a) < precision:
                break
            if f_c > f_d:
                b = d
                d = c
                c = b - phi * (b - a)
                f_d = f_c
                f_c = lyapunov(c)
            else:
                a = c
                c = d
                d = a + phi * (b - a)
                f_c = f_d
                f_d = lyapunov(d)
        results[i] = (a + b) / 2
    return results
bounds_to_test = np.array([[0.74, 0.76],[1.24,1.26]]) 
ans = refine_by_0618(bounds_to_test)
exact = np.array([0.75, 1.25])
print(ans)
print(ans-exact)
