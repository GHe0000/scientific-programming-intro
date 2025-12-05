
import numpy as np
import matplotlib.pyplot as plt
import boxdim

def circle(n=20000):
    theta = np.linspace(0, 2*np.pi, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])

def square_boundary(n=20000):
    t = np.linspace(0, 1, n)
    top = np.column_stack([t, np.ones_like(t)])
    bottom = np.column_stack([t, np.zeros_like(t)])
    left = np.column_stack([np.zeros_like(t), t])
    right = np.column_stack([np.ones_like(t), t])
    return np.vstack([top, bottom, left, right])

def filled_square(n=50000):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return np.column_stack([x, y])

def cantor_set(level=8):
    segments = [(0, 1)]
    for _ in range(level):
        new_segments = []
        for a, b in segments:
            third = (b - a) / 3
            new_segments.append((a, a+third))
            new_segments.append((b-third, b))
        segments = new_segments
    points = []
    for a, b in segments:
        x = np.linspace(a, b, 50)
        points.append(np.column_stack([x, np.zeros_like(x)]))
    return np.vstack(points)

def sierpinski_triangle(n=100000):
    verts = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]])
    p = np.random.rand(2)
    pts = []
    for _ in range(n):
        v = verts[np.random.randint(0,3)]
        p = (p + v) / 2
        pts.append(p)
    return np.array(pts)

def sierpinski_carpet(n=100000):
    p = np.random.rand(2)
    pts = []
    for _ in range(n):
        i, j = divmod(np.random.randint(0,8), 3)
        if (i,j) == (1,1):
            continue
        p = (p + np.array([i,j])) / 3
        pts.append(p)
    return np.array(pts)

def random_walk_1d(n=500000):
    steps = np.random.choice([-1, 1], size=n)
    path = np.cumsum(steps)
    t = np.linspace(0, 1, n)
    path_norm = (path - path.min()) / (path.max() - path.min())
    return np.column_stack([t, path_norm])

def analyze(points, name, theory_dim=None, m_list=None):
    if m_list is None:
        m_list = np.arange(100, 1000, 100)
    counts = [boxdim.count_occupied_boxes(points, m) for m in m_list]
    k, b = np.polyfit(np.log(m_list), np.log(counts), 1)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(m_list, counts, color="blue", label="Data")
    ax.plot(m_list, np.exp(k*np.log(m_list)+b), "r", label=f"Fit: D≈{k:.3f}")
    ax.set_xlabel("m (divisions)")
    ax.set_ylabel("Occupied boxes")
    if theory_dim:
        ax.set_title(f"{name} (Theory D={theory_dim:.3f})")
    else:
        ax.set_title(f"{name}")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    boxdim.visualize_boxes(points, m=30, ax=ax, facecolor="skyblue", alpha=0.4, edgecolor="gray")
    ax.plot(points[:,0], points[:,1], "k.", markersize=1)
    ax.set_aspect("equal")
    ax.set_title(f"{name} with boxes (m=30)")
    plt.show()

    return k


# ==== 测试 ====
# analyze(circle(), "Circle", theory_dim=1.0)
# analyze(square_boundary(), "Square boundary", theory_dim=1.0)
# analyze(filled_square(), "Filled square", theory_dim=2.0)
# analyze(cantor_set(), "Cantor set", theory_dim=np.log(2)/np.log(3))
# analyze(sierpinski_triangle(), "Sierpinski triangle", theory_dim=np.log(3)/np.log(2))
# analyze(sierpinski_carpet(), "Sierpinski carpet", theory_dim=np.log(8)/np.log(3))
analyze(random_walk_1d(), "1D Random Walk", theory_dim=1.5)
