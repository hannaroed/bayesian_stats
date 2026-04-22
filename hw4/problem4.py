import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### 4e) ###

df = pd.read_csv("hw4/data/wagedata.csv")

x = df["Exper"].to_numpy(dtype=float)
y = np.log(df["WeeklyEarnings"].to_numpy(dtype=float))
n = len(y)
p = 3

def fit_for_c(c, x, y):
    Xc = np.column_stack([
        np.ones(len(x)),
        x,
        np.maximum(x - c, 0.0)
    ])
    XtX = Xc.T @ Xc
    Xty = Xc.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)
    resid = y - Xc @ beta_hat
    S = resid @ resid
    detXtX = np.linalg.det(XtX)
    return beta_hat, S, detXtX, Xc

# grid of c values
c_grid = np.linspace(2.01, 59.99, 1000)

log_unnorm_post = np.empty_like(c_grid)

for j, c in enumerate(c_grid):
    beta_hat_c, S_c, detXtX_c, _ = fit_for_c(c, x, y)
    log_unnorm_post[j] = -0.5*np.log(detXtX_c) - ((n-p)/2.0)*np.log(S_c)

# normalize stably
m = np.max(log_unnorm_post)
unnorm = np.exp(log_unnorm_post - m)
post_c = unnorm / np.sum(unnorm)

print("Posterior mean of c:", np.sum(c_grid * post_c))
print("Posterior mode of c:", c_grid[np.argmax(post_c)])

cdf = np.cumsum(post_c)

lower_idx = np.searchsorted(cdf, 0.025)
upper_idx = np.searchsorted(cdf, 0.975)

c_lower = c_grid[lower_idx]
c_upper = c_grid[upper_idx]

print(f"95% uncertainty interval for c: ({float(c_lower):.4f}, {float(c_upper):.4f})")

plt.figure(figsize=(8,5))
plt.plot(c_grid, post_c)
plt.xlabel("c")
plt.ylabel("Posterior probability (discrete approximation)")
plt.title("Posterior of c")
plt.show()

### 4f) ###
rng = np.random.default_rng(42)

N = 100
samples = []

for _ in range(N):
    # sample c from discrete posterior
    c_samp = rng.choice(c_grid, p=post_c)

    beta_hat_c, S_c, detXtX_c, Xc = fit_for_c(c_samp, x, y)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)

    # sample sigma^2 from inverse-gamma
    alpha = (n - p) / 2.0
    rate = S_c / 2.0
    tau = rng.gamma(shape=alpha, scale=1.0/rate) # tau = 1/sigma^2
    sigma2_samp = 1.0 / tau
    sigma_samp = np.sqrt(sigma2_samp)

    beta_samp = rng.multivariate_normal(beta_hat_c, sigma2_samp * XtX_inv)

    samples.append((beta_samp, c_samp, sigma_samp))

print("Number of samples:", len(samples))

x_grid = np.linspace(x.min(), x.max(), 300)

plt.figure(figsize=(10,6))
plt.scatter(x, y, s=3, alpha=0.25)

for beta_samp, c_samp, sigma_samp in samples:
    curve = beta_samp[0] + beta_samp[1]*x_grid + beta_samp[2]*np.maximum(x_grid - c_samp, 0.0)
    plt.plot(x_grid, curve, alpha=0.2)

plt.xlabel("Experience")
plt.ylabel("log(Weekly Earnings)")
plt.title("100 posterior sampled broken-stick curves")
plt.show()
