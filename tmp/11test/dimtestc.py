import numpy as np
import matplotlib.pyplot as plt

def correlation_dim(pts, n_samp=10000, n_bins=20, plot=False):
    N = pts.shape[0]
    if n_samp <= N:
        samp_idx = np.random.choice(N, n_samp, replace=False)
        samp_pts = pts[samp_idx]
    else:
        samp_pts = pts

    pts_sq = np.sum(pts**2, axis=1)
    samp_pts_sq = np.sum(samp_pts**2, axis=1)
    dist_sq = samp_pts_sq[:, np.newaxis] + pts_sq[np.newaxis, :] - 2 * samp_pts @ pts.T
    dist_sq = np.maximum(dist_sq, 0)
    dist = np.sqrt(dist_sq.ravel())
    dist = dist[dist > 1e-10]
    r_min, r_max = np.min(dist), np.max(dist)
    bins = np.geomspace(r_min, r_max, n_bins + 1)
    counts, _ = np.histogram(dist, bins=bins)
    C_r = np.cumsum(counts)
    C_r = C_r / C_r[-1]
    radius = bins[1:]
    log_r = np.log(radius)
    log_Cr = np.log(C_r)
    slope, intercept = np.polyfit(log_r, log_Cr, 1)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(log_r, log_Cr, 'o', markersize=3, label='Data', alpha=0.5)
        plt.plot(log_r, slope * log_r + intercept, 'r-', lw=2, label=f'Fit: D2={slope:.3f}')
        plt.xlabel(r'$\ln(r)$')
        plt.ylabel(r'$\ln(C(r))$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    return slope

def get_circle_3d(n=3000, r=10.0):
    theta = np.linspace(0, 2*np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z])

def get_random_walk_1d(n=3000):
    steps = np.random.choice([-1, 1], size=n)
    x = np.cumsum(steps)
    t = np.linspace(0, 1, n)
    x_norm = (x - x.min()) / (x.max() - x.min())
    return np.column_stack([t, x_norm])

circle_pts = get_circle_3d()
d2_circle = correlation_dim(circle_pts, plot=True)
print(f"Result: {d2_circle:.4f}\n")

rw_pts = get_random_walk_1d()
d2_rw = correlation_dim(rw_pts, plot=True)
print(f"Result: {d2_rw:.4f}")
