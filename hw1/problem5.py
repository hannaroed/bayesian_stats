import numpy as np
import math

x = np.array([26.6, 38.5, 34.4, 34.0, 31.0, 23.6], float)
n = x.size
a, b = 23.0, 39.0

SQRT2 = math.sqrt(2.0)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

def Phi(z): return 0.5 * (1.0 + math.erf(z / SQRT2))

def log_denom(u, v):
    d = Phi(u) - Phi(v)
    return -np.inf if d <= 0 else math.log(d)

### 5(c) ###
theta = np.linspace(15.0, 50.0, 701)
logsig = np.linspace(-1.0, 3.0, 801)
dtheta = theta[1] - theta[0]
dlog = logsig[1] - logsig[0]

logW = np.full((theta.size, logsig.size), -np.inf)
for i, th in enumerate(theta):
    for j, ls in enumerate(logsig):
        sig = math.exp(ls)
        ld = log_denom((b - th) / sig, (a - th) / sig)
        if not np.isfinite(ld): 
            continue
        z = (x - th) / sig
        s_logphi = float(np.sum(-LOG_SQRT_2PI - 0.5 * z * z))
        logW[i, j] = (-n * ls) + s_logphi - (n * ld) # posterior in (theta, log_sigma) up to const

mx = np.max(logW[np.isfinite(logW)])
W = np.exp(logW - mx)
Z = np.sum(W) * dtheta * dlog
Wn = W / Z
print("Normalization check:", float(np.sum(Wn) * dtheta * dlog))

### 5(d) ###
p_th = np.sum(Wn, axis=1) * dlog
p_th /= (np.sum(p_th) * dtheta)
cdf = np.cumsum(p_th) * dtheta

def q_from_cdf(grid, c, q):
    k = np.searchsorted(c, q)
    if k <= 0: return float(grid[0])
    if k >= len(grid): return float(grid[-1])
    x0, x1 = grid[k-1], grid[k]
    c0, c1 = c[k-1], c[k]
    t = 0.0 if c1 == c0 else (q - c0) / (c1 - c0)
    return float(x0 + t * (x1 - x0))

lo, hi = q_from_cdf(theta, cdf, 0.025), q_from_cdf(theta, cdf, 0.975)
print("Bayesian credible interval (95%) for theta:", (lo, hi))

# Usual (non-truncated) interval
xbar = float(np.mean(x))
s = float(np.std(x, ddof=1))
t975 = 2.570581836
lo_u = xbar - t975 * s / math.sqrt(n)
hi_u = xbar + t975 * s / math.sqrt(n)
print("Usual (non-truncated) t 95% interval for theta:", (lo_u, hi_u))

