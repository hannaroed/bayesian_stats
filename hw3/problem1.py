import numpy as np
import pandas as pd
from scipy.stats import chi2

x = pd.read_csv("hw3/data/old_faithful_eruptions.txt", header=None, sep=r"\s+").values.flatten()
x = x.astype(float)
n = len(x)

# sample SD
s = np.std(x, ddof=1)

# classical bootstrap interval
B = 10000
rng = np.random.default_rng(238)

boot_sds = np.empty(B)
for b in range(B):
    x_star = rng.choice(x, size=n, replace=True)
    boot_sds[b] = np.std(x_star, ddof=1)

ci_boot = np.quantile(boot_sds, [0.025, 0.975])

# bayesian bootstrap interval
bb_sds = np.empty(B)
for b in range(B):
    w = rng.dirichlet(np.ones(n))
    mu_w = np.sum(w * x)
    var_w = np.sum(w * (x - mu_w)**2)
    bb_sds[b] = np.sqrt(var_w)

ci_bayes_boot = np.quantile(bb_sds, [0.025, 0.975])

# normal-theory exact interval

chi2_lower = chi2.ppf(0.025, df=n-1)
chi2_upper = chi2.ppf(0.975, df=n-1)

# CI for sigma
ci_normal = np.array([
    np.sqrt((n - 1) * s**2 / chi2_upper),
    np.sqrt((n - 1) * s**2 / chi2_lower)
])

print(f"n = {n}")
print(f"Sample SD = {s:.6f}\n")

print("95% interval for sigma")
print(f"Classical bootstrap: {ci_boot}")
print(f"Bayesian bootstrap:  {ci_bayes_boot}")
print(f"Normal-theory:       {ci_normal}")

# Comparing widths
widths = {
    "Classical bootstrap": ci_boot[1] - ci_boot[0],
    "Bayesian bootstrap": ci_bayes_boot[1] - ci_bayes_boot[0],
    "Normal-theory": ci_normal[1] - ci_normal[0],
}

print("\nInterval widths:")
for k, v in widths.items():
    print(f"{k}: {v:.6f}")

widest = max(widths, key=widths.get)
print(f"\nWidest interval: {widest}")
