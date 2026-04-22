import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### 7d) ###

df = pd.read_csv("hw5/data/wagedata.csv")

y = np.log(df["WeeklyEarnings"].to_numpy())
x = df["Exper"].to_numpy().astype(float)

n = len(y)
m = int(x.max()) # 63

# Design matrix: [1, x, ReLU(x-1), ..., ReLU(x-(m-1))]
X_cols = [np.ones(n), x]
for knot in range(1, m):
    X_cols.append(np.maximum(x - knot, 0.0))
X = np.column_stack(X_cols)

p = X.shape[1] # m+1 = 64

# J = diag(0,0,1,...,1)
J = np.diag(np.r_[0, 0, np.ones(m - 1)])

XtX = X.T @ X
Xty = X.T @ y
yty = y @ y

def gibbs_sampler(XtX, Xty, yty, J, m, n, n_keep=5000, burn=1000, seed=238):
    rng = np.random.default_rng(seed)
    p = XtX.shape[0]

    beta = np.zeros(p)
    gamma = 0.02
    sigma = np.std(y)

    beta_draws = np.zeros((n_keep, p))
    gamma_draws = np.zeros(n_keep)
    sigma_draws = np.zeros(n_keep)

    total_iters = burn + n_keep

    for t in range(total_iters):
        # beta update
        A = XtX + (gamma ** -2) * J
        Ainv = np.linalg.inv(A)
        beta_mean = Ainv @ Xty
        beta_cov = sigma**2 * Ainv
        beta = rng.multivariate_normal(beta_mean, beta_cov)

        # w = 1/gamma^2 update
        beta_pen_sq = np.sum(beta[2:] ** 2)
        shape_w = (m - 1) / 2
        rate_w = beta_pen_sq / (2 * sigma**2)
        w = rng.gamma(shape=shape_w, scale=1 / rate_w)
        gamma = 1 / np.sqrt(w)

        # eta = 1/sigma^2 update
        resid_quad = yty - 2 * beta @ Xty + beta @ XtX @ beta
        shape_eta = (n + m - 1) / 2
        rate_eta = 0.5 * (resid_quad + w * beta_pen_sq)
        eta = rng.gamma(shape=shape_eta, scale=1 / rate_eta)
        sigma = 1 / np.sqrt(eta)

        if t >= burn:
            idx = t - burn
            beta_draws[idx] = beta
            gamma_draws[idx] = gamma
            sigma_draws[idx] = sigma

    return beta_draws, gamma_draws, sigma_draws


beta_draws, gamma_draws, sigma_draws = gibbs_sampler(
    XtX=XtX, Xty=Xty, yty=yty, J=J, m=m, n=n,
    n_keep=5000, burn=1000, seed=238
)

beta_mean_gibbs = beta_draws.mean(axis=0)

x_grid = np.arange(0, m + 1, dtype=float)
X_grid_cols = [np.ones(len(x_grid)), x_grid]
for knot in range(1, m):
    X_grid_cols.append(np.maximum(x_grid - knot, 0.0))
X_grid = np.column_stack(X_grid_cols)

fit_gibbs = X_grid @ beta_mean_gibbs

gamma_grid = np.logspace(np.log10(1e-4), np.log10(10), 2000)
logpost_gamma = np.zeros(len(gamma_grid))

log_gamma_grid = np.log(gamma_grid)
dlog = log_gamma_grid[1] - log_gamma_grid[0]

for i, gamma in enumerate(gamma_grid):
    A = XtX + (gamma ** -2) * J
    Ainv = np.linalg.inv(A)
    sign, logdet_Ainv = np.linalg.slogdet(Ainv)
    sse_gamma = yty - Xty @ Ainv @ Xty
    logpost_gamma[i] = (-m) * np.log(gamma) + 0.5 * logdet_Ainv - (n / 2 - 1) * np.log(sse_gamma)

log_weights = logpost_gamma + np.log(gamma_grid) + np.log(dlog)
weights = np.exp(log_weights - np.max(log_weights))
weights = weights / np.sum(weights)

gamma_samples_lecture = np.random.default_rng(123).choice(
    gamma_grid, size=2000, replace=True, p=weights
)

sigma_samples_lecture = np.zeros(2000)
beta_draws_lecture = np.zeros((2000, p))
rng = np.random.default_rng(12345)

for i, gamma in enumerate(gamma_samples_lecture):
    A = XtX + (gamma ** -2) * J
    Ainv = np.linalg.inv(A)

    alpha_sigma = n / 2 - 1
    rate_sigma = (yty - Xty @ Ainv @ Xty) / 2
    sigma = np.sqrt(1 / rng.gamma(shape=alpha_sigma, scale=1 / rate_sigma))
    sigma_samples_lecture[i] = sigma

    beta_mean = Ainv @ Xty
    beta_cov = sigma**2 * Ainv
    beta_draws_lecture[i] = rng.multivariate_normal(beta_mean, beta_cov)

beta_mean_lecture = beta_draws_lecture.mean(axis=0)
fit_lecture = X_grid @ beta_mean_lecture

# Comparison
print("\nPosterior summaries:")
print("Gibbs gamma mean =", gamma_draws.mean(), "sd =", gamma_draws.std(ddof=1))
print("Lecture gamma mean =", gamma_samples_lecture.mean(), "sd =", gamma_samples_lecture.std(ddof=1))
print("Gibbs sigma mean =", sigma_draws.mean(), "sd =", sigma_draws.std(ddof=1))
print("Lecture sigma mean =", sigma_samples_lecture.mean(), "sd =", sigma_samples_lecture.std(ddof=1))

rmse_fit = np.sqrt(np.mean((fit_gibbs - fit_lecture) ** 2))
max_abs_fit = np.max(np.abs(fit_gibbs - fit_lecture))

print("\nComparison of fitted curves:")
print("RMSE between Gibbs and Lecture-20 posterior mean fits =", rmse_fit)
print("Max abs difference =", max_abs_fit)

plt.figure(figsize=(10, 6))
plt.plot(x_grid, fit_gibbs, linewidth=2, label="Gibbs posterior mean")
plt.plot(x_grid, fit_lecture, linewidth=2, linestyle="--", label="Lecture 20 method")
plt.xlabel("Experience")
plt.ylabel("log(WeeklyEarnings)")
plt.title("Gibbs vs Lecture 20 posterior mean fit")
plt.legend()
plt.show()
