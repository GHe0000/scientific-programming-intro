import numpy as np

def find_point(check_func, r_arr, tol=1e-6, max_step=50):
    a = r_arr[:, 0].copy()
    b = r_arr[:, 1].copy()

    for _ in range(max_step):
        mid = 0.5 * (a + b)
        check = check_func(mid)
        check = np.asarray(check, dtype=bool)

        a = np.where(check, a, mid)
        b = np.where(check, mid, b)

        if np.all(np.abs(b - a) < tol):
            break

    return 0.5 * (a + b)


def make_check_func(r_arr, tol=1e-6, max_iter=500):
    r_arr = np.atleast_1d(r_arr)
    x_plus = (-1 + np.sqrt(1 + 4*r_arr)) / 2.0  # 周期一不动点

    def check_func(x0):
        x = x0.copy()
        for _ in range(max_iter):
            x = r_arr - x*x
        return np.abs(x - x_plus) < tol
    return check_func


if __name__ == "__main__":
    r_vals = np.array([0.2, 0.5, 1.0, 1.5])
    x_plus = (-1 + np.sqrt(1 + 4*r_vals)) / 2
    T = (1 + np.sqrt(1 + 4*r_vals)) / 2

    intervals = np.vstack([x_plus, T]).T
    check_func = make_check_func(r_vals)
    boundaries = find_point(check_func, intervals, tol=1e-6, max_step=50)

    for r, b in zip(r_vals, boundaries):
        print(f"r={r:.2f}, basin boundary ≈ {b:.6f}")
