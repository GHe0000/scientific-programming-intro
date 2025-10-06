import numpy as np
import matplotlib.pyplot as plt

def find_point(check_func, r_arr, tol=1e-6, max_step=50):
    a = r_arr[:, 0].copy()
    b = r_arr[:, 1].copy()
    for _ in range(max_step):
        mid = 0.5 * (a + b)
        check = check_func(mid)
        check = np.asarray(check, dtype=bool)
        a = np.where(check, mid, a)
        b = np.where(check, b, mid)
        if np.all(np.abs(b - a) < tol):
            break
    return 0.5 * (a + b)

def make_classifier(r, tol=1e-6, max_step=500):
    x_star1 = (1 + np.sqrt(4*r-3)) / 2.0
    x_star2 = (1 - np.sqrt(4*r-3)) / 2.0
    def classify(x0):
        x = x0.copy()
        for _ in range(max_step):
            x = r - x**2
        return np.where(np.abs(x-x_star1) < np.abs(x-x_star2), 0, 1)
    return classify

def find_boundaries_single(r, n_grid=2000):
    domain = (-(np.sqrt(4*r+1)+1)/2, (np.sqrt(4*r+1)+1)/2)
    classify = make_classifier(r)
    
    xs = np.linspace(domain[0], domain[1], n_grid)
    labels = classify(xs)

    change_idx = np.where(labels[:-1] != labels[1:])[0]
    boundaries = []
    for idx in change_idx:
        a, b = xs[idx], xs[idx+1]
        def check_func(mid):
            return classify(np.array([a]))[0] != classify(np.array([mid]))[0]
        root = find_point(lambda mid: [check_func(mid[0])], np.array([[a,b]]))
        boundaries.append(root[0])
    return np.array(boundaries)

def find_boundaries_batch(r_arr, n_grid=2000):
    results = []
    for r in r_arr:
        try:
            bounds = find_boundaries_single(r, n_grid=n_grid)
        except Exception:
            bounds = np.array([])
        results.append(bounds)
    return results

def plot_boundaries(r_arr, results):
    plt.figure(figsize=(8,6))
    for r, bounds in zip(r_arr, results):
        plt.scatter([r]*len(bounds), bounds, c="k", s=5)
    plt.xlabel("r")
    plt.ylabel("分界点位置")
    plt.title("周期二不同轨道的吸引域分界点随 r 的变化")
    plt.grid(alpha=0.3)
    plt.show()

# 示例
r_arr = np.linspace(0.8, 1.25, 200)
results = find_boundaries_batch(r_arr, n_grid=1000)
plot_boundaries(r_arr, results)
