import numpy as np
import math
import matplotlib.pyplot as plt

def q_from_cdf(grid, c, q):
    k = np.searchsorted(c, q)
    if k <= 0: return float(grid[0])
    if k >= len(grid): return float(grid[-1])
    x0, x1 = grid[k-1], grid[k]
    c0, c1 = c[k-1], c[k]
    t = 0.0 if c1 == c0 else (q - c0) / (c1 - c0)
    return float(x0 + t * (x1 - x0))

### 7(b) ###

x = np.array([26.6, 38.5, 34.4, 34.0, 31.0, 23.6], float)
n = x.size

theta = np.linspace(15.0, 50.0, 2001)
dtheta = theta[1] - theta[0]

M = np.array([np.sum(np.abs(x - th)) for th in theta])
w = M ** (-n)
p = w / (np.sum(w) * dtheta)

print("Normalization check:", float(np.sum(p) * dtheta))

plt.figure()
plt.plot(theta, p)
plt.xlabel("theta")
plt.ylabel("posterior density")
plt.title("Posterior of theta (Laplace model)")
plt.show()

### 7(c) ###

cdf = np.cumsum(p) * dtheta
lo = q_from_cdf(theta, cdf, 0.025)
hi = q_from_cdf(theta, cdf, 0.975)
print("Bayesian credible interval (95%) for theta:", (lo, hi))

xbar = float(np.mean(x))
s = float(np.std(x, ddof=1))
t975_df5 = 2.570581836
lo_u = xbar - t975_df5 * s / math.sqrt(n)
hi_u = xbar + t975_df5 * s / math.sqrt(n)
print("Usual t interval (95%) for theta:", (lo_u, hi_u))

### 7(d) ###

x2 = np.array([26.6, 38.5, 34.4, 34.0, 31.0, 23.6, 120.0], float)
n2 = x2.size

theta2 = np.linspace(15.0, 130.0, 4001)
dtheta2 = theta2[1] - theta2[0]

M2 = np.array([np.sum(np.abs(x2 - th)) for th in theta2])
w2 = M2 ** (-n2)
p2 = w2 / (np.sum(w2) * dtheta2)
cdf2 = np.cumsum(p2) * dtheta2

lo2 = q_from_cdf(theta2, cdf2, 0.025)
hi2 = q_from_cdf(theta2, cdf2, 0.975)
print("Bayesian credible interval (95%) for theta (with outlier):", (lo2, hi2))

xbar2 = float(np.mean(x2))
s2 = float(np.std(x2, ddof=1))
t975_df6 = 2.446911851
lo_u2 = xbar2 - t975_df6 * s2 / math.sqrt(n2)
hi_u2 = xbar2 + t975_df6 * s2 / math.sqrt(n2)
print("Usual t interval (95%) for theta (with outlier):", (lo_u2, hi_u2))