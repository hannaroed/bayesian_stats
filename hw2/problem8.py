import numpy as np
from scipy.special import gammaln

names = np.array([
    "Clemente", "F Robinson", "F Howard", "Johnstone", "Berry", "Spencer",
    "Kessinger", "L Alvarado", "Santo", "Swoboda", "Unser", "Williams",
    "Scott", "Petrocelli", "E Rodriguez", "Campaneris", "Munson", "Alvis",
])

h = np.array([18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7], dtype=float)
theta_true = np.array([
    0.346, 0.298, 0.276, 0.222, 0.273, 0.270, 0.263, 0.210, 0.269,
    0.230, 0.264, 0.256, 0.303, 0.264, 0.226, 0.286, 0.316, 0.200,
], dtype=float)

n = 45.0
N = len(h)
rng = np.random.default_rng(238)

def log_marginal(a, b):
    return float(np.sum(
        gammaln(a + h)
        + gammaln(b + n - h)
        - gammaln(a + b + n)
        - gammaln(a)
        - gammaln(b)
        + gammaln(a + b)
    ))

### 8(a) ###

log_a_grid = np.linspace(np.log(1.0), np.log(1500.0), 220)
log_b_grid = np.linspace(np.log(1.0), np.log(4000.0), 240)
a_grid = np.exp(log_a_grid)
b_grid = np.exp(log_b_grid)

logw = np.zeros((len(a_grid), len(b_grid)))
for i, a in enumerate(a_grid):
    for j, b in enumerate(b_grid):
        logw[i, j] = log_marginal(a, b)

m = np.max(logw)
w = np.exp(logw - m)
w = w / np.sum(w)

theta_bayes = np.zeros(N)
for i, a in enumerate(a_grid):
    for j, b in enumerate(b_grid):
        theta_bayes += w[i, j] * ((a + h) / (a + b + n))

S = 30000
flat_w = w.ravel()
draw_idx = rng.choice(flat_w.size, size=S, p=flat_w)
ia, ib = np.unravel_index(draw_idx, w.shape)
a_s = a_grid[ia]
b_s = b_grid[ib]

theta_samples = rng.beta(
    a_s[:, None] + h[None, :],
    b_s[:, None] + (n - h)[None, :],
)

theta_naive = h / n

x = np.arcsin(np.sqrt((h + 0.25) / (n + 0.5)))
sigma2 = 1.0 / (4.0 * n)
xbar = np.mean(x)
ss = np.sum((x - xbar) ** 2)
shrink = max(0.0, 1.0 - (N - 3.0) * sigma2 / ss)
x_js = xbar + shrink * (x - xbar)
theta_js = np.sin(x_js) ** 2

def sse(est):
    return float(np.sum((est - theta_true) ** 2))

def sae(est):
    return float(np.sum(np.abs(est - theta_true)))

print("SSE (Bayes) =", sse(theta_bayes))
print("SSE (Naive) =", sse(theta_naive))
print("SSE (James-Stein) =", sse(theta_js))
print("SAE (Bayes) =", sae(theta_bayes))
print("SAE (Naive) =", sae(theta_naive))
print("SAE (James-Stein) =", sae(theta_js))

### 8(b) ###

ci_lo = np.quantile(theta_samples, 0.025, axis=0)
ci_hi = np.quantile(theta_samples, 0.975, axis=0)
inside = (theta_true >= ci_lo) & (theta_true <= ci_hi)

print("95% CI coverage count =", int(np.sum(inside)), "out of", N)
