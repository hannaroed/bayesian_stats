import numpy as np
import pandas as pd

# load the data
df = pd.read_csv("hw3/data/PearsonHeightData.txt", sep=r"\s+|,|\t", engine="python", header=0)

x = df.iloc[:, 0].to_numpy(dtype=float)
y = df.iloc[:, 1].to_numpy(dtype=float)
n = len(x)

# sample correlation
r_hat = np.corrcoef(x, y)[0, 1]

B = 10000
rng = np.random.default_rng(238)

# classical bootstrap
boot_corr = np.empty(B)

for b in range(B):
    idx = rng.choice(n, size=n, replace=True)
    xb = x[idx]
    yb = y[idx]
    boot_corr[b] = np.corrcoef(xb, yb)[0, 1]

ci_classical = np.quantile(boot_corr, [0.025, 0.975])

# bayesian bootstrap
bayes_corr = np.empty(B)

for b in range(B):
    w = rng.dirichlet(np.ones(n))
    
    xw = np.sum(w * x)
    yw = np.sum(w * y)
    
    cov_w = np.sum(w * (x - xw) * (y - yw))
    var_xw = np.sum(w * (x - xw)**2)
    var_yw = np.sum(w * (y - yw)**2)
    
    bayes_corr[b] = cov_w / np.sqrt(var_xw * var_yw)

ci_bayes = np.quantile(bayes_corr, [0.025, 0.975])

print(f"Sample correlation = {r_hat:.6f}")
print(f"Classical bootstrap 95% CI for rho: {ci_classical}")
print(f"Bayesian bootstrap 95% CI for rho:  {ci_bayes}")

print("\nWidths:")
print(f"Classical bootstrap width: {ci_classical[1] - ci_classical[0]:.6f}")
print(f"Bayesian bootstrap width:  {ci_bayes[1] - ci_bayes[0]:.6f}")
