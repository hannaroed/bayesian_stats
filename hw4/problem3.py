import numpy as np
import pandas as pd
import statsmodels.api as sm

### 3b) ###

height = pd.read_csv("hw4/data/PearsonHeightData.txt", sep=r"\s+")

x = height["Father"].to_numpy()
y = height["Son"].to_numpy()
n = len(y)

b1_coarse = np.linspace(0.40, 0.60, 801)
best_S = np.inf
best_b0 = None
best_b1 = None

for b1 in b1_coarse:
    b0 = np.median(y - b1 * x)
    S = np.sum(np.abs(y - b0 - b1 * x))
    if S < best_S:
        best_S = S
        best_b0 = b0
        best_b1 = b1

print("Center of grid (approximate LAD fit):")
print(f"beta0_center = {best_b0:.4f}")
print(f"beta1_center = {best_b1:.4f}")
print(f"min S        = {best_S:.4f}")

beta0_grid = np.linspace(best_b0 - 6, best_b0 + 6, 401)
beta1_grid = np.linspace(best_b1 - 0.08, best_b1 + 0.08, 401)

S_grid = np.empty((len(beta0_grid), len(beta1_grid)))

for j, b1 in enumerate(beta1_grid):
    fitted = y - b1 * x
    residuals = fitted[:, None] - beta0_grid[None, :]
    S_grid[:, j] = np.sum(np.abs(residuals), axis=0)

B0, B1 = np.meshgrid(beta0_grid, beta1_grid, indexing="ij")

logw = -n * np.log(S_grid)
logw -= np.max(logw)
w = np.exp(logw)
w /= np.sum(w)

beta0_mean = np.sum(B0 * w)
beta1_mean = np.sum(B1 * w)

beta0_sd = np.sqrt(np.sum((B0 - beta0_mean) ** 2 * w))
beta1_sd = np.sqrt(np.sum((B1 - beta1_mean) ** 2 * w))

mode_index = np.unravel_index(np.argmax(w), w.shape)
beta0_mode = B0[mode_index]
beta1_mode = B1[mode_index]

print("\nPosterior summaries from the discrete grid:")
print(f"Posterior mean beta0 = {beta0_mean:.4f}")
print(f"Posterior mean beta1 = {beta1_mean:.4f}")
print(f"Posterior sd   beta0 = {beta0_sd:.4f}")
print(f"Posterior sd   beta1 = {beta1_sd:.4f}")

print("\nPosterior mode (grid maximum):")
print(f"Posterior mode beta0 = {beta0_mode:.4f}")
print(f"Posterior mode beta1 = {beta1_mode:.4f}")

### 3c) ###

X = np.column_stack([np.ones(n), x])
ols_fit = sm.OLS(y, X).fit()

ols_intercept = ols_fit.params[0]
ols_slope = ols_fit.params[1]
ols_se_intercept = ols_fit.bse[0]
ols_se_slope = ols_fit.bse[1]

print("\nOLS results:")
print(f"Intercept estimate = {ols_intercept:.4f}")
print(f"Slope estimate     = {ols_slope:.4f}")
print(f"SE(intercept)      = {ols_se_intercept:.4f}")
print(f"SE(slope)          = {ols_se_slope:.4f}")
