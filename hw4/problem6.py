import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("hw4/data/GolfTrends13March2026.csv")
df["Time"] = pd.to_datetime(df["Time"])

y = df["golf"].to_numpy(dtype=float)
dates = df["Time"]
n = len(y)
t = np.arange(1, n + 1)

# build design matrix
X = np.zeros((n, n + 2))
X[:, 0] = 1.0
X[:, 1] = t - 1
for j in range(2, n):
    X[:, j] = np.maximum(t - j, 0)
X[:, n] = np.cos(2 * np.pi * t / 12)
X[:, n + 1] = np.sin(2 * np.pi * t / 12)

# penalty only on beta_2, ..., beta_{n-1}
penalty = np.zeros(n + 2)
penalty[2:n] = 1.0


def ridge_fit(X_train, y_train, lam):
    A = X_train.T @ X_train + lam * np.diag(penalty)
    b = X_train.T @ y_train
    return np.linalg.solve(A, b)


def forward_cv(X, y, lambdas, initial_train=120, horizon=12, step=12):
    cv_mse = []
    starts = list(range(initial_train, len(y) - horizon + 1, step))

    for lam in lambdas:
        fold_errors = []
        for start in starts:
            beta_hat = ridge_fit(X[:start], y[:start], lam)
            pred = X[start:start + horizon] @ beta_hat
            mse = np.mean((y[start:start + horizon] - pred) ** 2)
            fold_errors.append(mse)
        cv_mse.append(np.mean(fold_errors))

    return np.array(cv_mse)

### 6a) ###

lambdas = np.logspace(3, 5, 161)
cv_mse = forward_cv(X, y, lambdas)

best_idx = np.argmin(cv_mse)
lambda_hat = lambdas[best_idx]

beta_ridge = ridge_fit(X, y, lambda_hat)

mu_ridge = X[:, :n] @ beta_ridge[:n]
fit_ridge = X @ beta_ridge

print("lambda =", lambda_hat)
print("cv mse =", cv_mse[best_idx])

plt.figure(figsize=(11, 5))
plt.plot(dates, y, label="Data", alpha=0.7)
plt.plot(dates, mu_ridge, label="Estimated trend", linewidth=2)
plt.plot(dates, fit_ridge, label="Trend + seasonal part", linewidth=2)
plt.title("Ridge estimates")
plt.xlabel("Date")
plt.ylabel("Google Trends: golf")
plt.legend()
plt.tight_layout()
plt.show()

### 6b) (i) ###

# unpenalized part: intercept, slope, cosine, sine
X0 = np.column_stack([X[:, 0], X[:, 1], X[:, n], X[:, n + 1]])

# penalized part: hinge coefficients
Z = X[:, 2:n]

Q0, _ = np.linalg.qr(X0)
M0 = np.eye(n) - Q0 @ Q0.T

y_tilde = M0 @ y
Z_tilde = M0 @ Z

B = Z_tilde @ Z_tilde.T
eigvals, U = np.linalg.eigh(B)
alpha = U.T @ y_tilde
m = n - 4

log_gamma_grid = np.linspace(-10, -2, 250)
gamma_grid = np.exp(log_gamma_grid)

log_sigma_grid = np.linspace(-2, 5, 250)
sigma_grid = np.exp(log_sigma_grid)

log_post = np.empty((len(gamma_grid), len(sigma_grid)))

for i, gamma in enumerate(gamma_grid):
    denom = 1.0 + (gamma ** 2) * eigvals
    log_det = np.sum(np.log(denom))
    quad = np.sum((alpha ** 2) / denom)

    log_post[i, :] = (
        -0.5 * log_det
        - (m + 1) * np.log(sigma_grid)
        - quad / (2.0 * sigma_grid ** 2)
        - np.log(gamma)
    )

d_gamma = np.gradient(gamma_grid)
d_sigma = np.gradient(sigma_grid)

weights = np.exp(log_post - log_post.max()) * d_gamma[:, None] * d_sigma[None, :]
weights = weights / weights.sum()

gamma_hat = np.sum(weights * gamma_grid[:, None])
sigma_hat = np.sum(weights * sigma_grid[None, :])

rng = np.random.default_rng(1)
flat_idx = rng.choice(weights.size, size=1000, p=weights.ravel())
g_idx, s_idx = np.unravel_index(flat_idx, weights.shape)

gamma_samps = gamma_grid[g_idx]
sigma_samps = sigma_grid[s_idx]

print("gamma hat =", gamma_hat)
print("sigma hat =", sigma_hat)
print("1 / gamma_hat^2 =", 1.0 / gamma_hat**2)
print("lambda =", lambda_hat)

### 6b) (ii) ###

rng = np.random.default_rng(2)

mu_samples = np.empty((1000, n))
mu_conditional_means = np.empty((1000, n))

for k, (gamma, sigma) in enumerate(zip(gamma_samps, sigma_samps)):
    prior_var = np.r_[[1e8, 1e8], np.repeat((gamma * sigma) ** 2, n - 2), [1e8, 1e8]]
    Dinv = np.diag(1.0 / prior_var)

    A = X.T @ X / (sigma ** 2) + Dinv
    rhs = X.T @ y / (sigma ** 2)

    L = np.linalg.cholesky(A)

    beta_mean = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    mu_conditional_means[k] = X[:, :n] @ beta_mean[:n]

    z = rng.normal(size=n + 2)
    beta_samp = beta_mean + np.linalg.solve(L.T, z)
    mu_samples[k] = X[:, :n] @ beta_samp[:n]

print("number of posterior samples =", mu_samples.shape[0])

plt.figure(figsize=(11, 5))
plt.plot(dates, y, label="Data", alpha=0.6)

for j in range(200):
    plt.plot(dates, mu_samples[j], alpha=0.03)

plt.title("Posterior trend samples")
plt.xlabel("Date")
plt.ylabel("Google Trends: golf")
plt.tight_layout()
plt.show()

### 6b) (iii) ###

mu_bayes = mu_conditional_means.mean(axis=0)

ridge_roughness = np.sum(np.diff(mu_ridge, 2) ** 2)
bayes_roughness = np.sum(np.diff(mu_bayes, 2) ** 2)

print("ridge roughness =", ridge_roughness)
print("bayes roughness =", bayes_roughness)

plt.figure(figsize=(11, 5))
plt.plot(dates, y, label="Data", alpha=0.7)
plt.plot(dates, mu_ridge, label="Ridge estimate", linewidth=2)
plt.plot(dates, mu_bayes, label="Bayesian estimate", linewidth=2)
plt.title("Trend estimates")
plt.xlabel("Date")
plt.ylabel("Google Trends: golf")
plt.legend()
plt.tight_layout()
plt.show()

### 6b) (iv) ###

rng = np.random.default_rng(2)

beta_season_means = np.empty((1000, 2))

for k, (gamma, sigma) in enumerate(zip(gamma_samps, sigma_samps)):
    prior_var = np.r_[[1e8, 1e8], np.repeat((gamma * sigma) ** 2, n - 2), [1e8, 1e8]]
    Dinv = np.diag(1.0 / prior_var)

    A = X.T @ X / (sigma ** 2) + Dinv
    rhs = X.T @ y / (sigma ** 2)

    L = np.linalg.cholesky(A)
    beta_mean = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

    beta_season_means[k] = beta_mean[n:n + 2]

beta_n_hat, beta_np1_hat = beta_season_means.mean(axis=0)

fit_bayes = (
    mu_bayes
    + beta_n_hat * np.cos(2 * np.pi * t / 12)
    + beta_np1_hat * np.sin(2 * np.pi * t / 12)
)

print("beta_n hat =", beta_n_hat)
print("beta_n+1 hat =", beta_np1_hat)

plt.figure(figsize=(11, 5))
plt.plot(dates, y, label="Data", alpha=0.7)
plt.plot(dates, fit_bayes, label="Posterior mean fit", linewidth=2)
plt.title("Bayesian fit with seasonal part")
plt.xlabel("Date")
plt.ylabel("Google Trends: golf")
plt.legend()
plt.tight_layout()
plt.show()
