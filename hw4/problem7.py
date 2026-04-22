import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("hw4/data/wagedata.csv")
df["log_weekly_earnings"] = np.log(df["WeeklyEarnings"])

def brownian_kernel(x, z):
    return np.minimum.outer(x, z)

def grouped_stats(data):
    g = (
        data.groupby("Exper")["log_weekly_earnings"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("Exper")
    )

    merged = data[["Exper", "log_weekly_earnings"]].merge(
        g[["Exper", "mean"]], on="Exper", how="left"
    )
    within_sse = np.sum((merged["log_weekly_earnings"] - merged["mean"]) ** 2)

    x = g["Exper"].to_numpy(dtype=float)
    ybar = g["mean"].to_numpy(dtype=float)
    nrep = g["count"].to_numpy(dtype=float)
    n_total = int(nrep.sum())

    return x, ybar, nrep, within_sse, n_total

def fixed_gamma_quantities(x, ybar, nrep, within_sse, gamma):
    K = brownian_kernel(x, x)
    A = (gamma ** 2) * K + np.diag(1.0 / nrep)

    one = np.ones(len(x))
    Ainv_one = np.linalg.solve(A, one)
    Ainv_y = np.linalg.solve(A, ybar)

    beta0_hat = (one @ Ainv_y) / (one @ Ainv_one)

    resid = ybar - beta0_hat * one
    quad_between = resid @ np.linalg.solve(A, resid)
    S_gamma = within_sse + quad_between

    sign, logdetA = np.linalg.slogdet(A)
    log_one_Ainv_one = np.log(one @ Ainv_one)

    return A, beta0_hat, S_gamma, logdetA, log_one_Ainv_one

def posterior_gamma_grid(x, ybar, nrep, within_sse, n_total, log_gamma_grid):
    gamma_grid = np.exp(log_gamma_grid)
    log_post = np.empty_like(gamma_grid)

    A_list = []
    beta0_list = []
    S_list = []

    for i, gamma in enumerate(gamma_grid):
        A, beta0_hat, S_gamma, logdetA, log_one_Ainv_one = fixed_gamma_quantities(
            x, ybar, nrep, within_sse, gamma
        )

        A_list.append(A)
        beta0_list.append(beta0_hat)
        S_list.append(S_gamma)

        log_post[i] = (
            -np.log(gamma)
            - 0.5 * logdetA
            - 0.5 * log_one_Ainv_one
            - 0.5 * (n_total - 1) * np.log(S_gamma)
        )

    w = np.exp(log_post - log_post.max())
    dgamma = np.gradient(gamma_grid)
    w = w * dgamma
    w = w / w.sum()

    gamma_hat = np.sum(w * gamma_grid)
    return gamma_grid, w, gamma_hat, A_list, beta0_list, S_list

def posterior_mean_f_given_gamma(x, ybar, x_grid, gamma, A, beta0_hat):
    K_star = brownian_kernel(x_grid, x)
    resid = ybar - beta0_hat
    alpha = np.linalg.solve(A, resid)
    return beta0_hat + (gamma ** 2) * K_star @ alpha

def posterior_mean_f_mixture(x, ybar, x_grid, gamma_grid, gamma_weights, A_list, beta0_list):
    f_post_mean = np.zeros_like(x_grid)

    for gamma, w, A, beta0_hat in zip(gamma_grid, gamma_weights, A_list, beta0_list):
        f_g = posterior_mean_f_given_gamma(x, ybar, x_grid, gamma, A, beta0_hat)
        f_post_mean += w * f_g

    return f_post_mean

def posterior_sigma_given_gamma(S_gamma, n_total, rng, size=1000):
    shape = (n_total - 1) / 2.0
    rate = S_gamma / 2.0
    sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / rate, size=size)
    return np.sqrt(sigma2)

### 7e) ###

sub = df.sample(n=500, random_state=42).copy()

x, ybar, nrep, within_sse, n_total = grouped_stats(sub)

log_gamma_grid = np.linspace(-8, 3, 400)
gamma_grid, gamma_weights, gamma_hat, A_list, beta0_list, S_list = posterior_gamma_grid(
    x, ybar, nrep, within_sse, n_total, log_gamma_grid
)

rng = np.random.default_rng(42)

idx_hat = np.argmin(np.abs(gamma_grid - gamma_hat))
sigma_samps = posterior_sigma_given_gamma(S_list[idx_hat], n_total, rng, size=1000)
sigma_hat = np.mean(sigma_samps)

x_grid = np.linspace(0, 63, 300)
f_post_mean = posterior_mean_f_mixture(
    x, ybar, x_grid, gamma_grid, gamma_weights, A_list, beta0_list
)

print("subsample size =", 500)
print("unique experience values =", len(x))
print("beta0 hat =", beta0_list[idx_hat])
print("gamma hat =", gamma_hat)
print("sigma hat =", sigma_hat)

plt.figure(figsize=(10, 5))
plt.scatter(sub["Exper"], sub["log_weekly_earnings"], s=12, alpha=0.35, label="Data")
plt.plot(x_grid, f_post_mean, linewidth=2, label="Posterior mean")
plt.title("Posterior mean estimate on subsample")
plt.xlabel("Experience")
plt.ylabel("Log weekly earnings")
plt.legend()
plt.tight_layout()
plt.show()

### 7f) ###

x, ybar, nrep, within_sse, n_total = grouped_stats(df)

gamma_grid, gamma_weights, gamma_hat, A_list, beta0_list, S_list = posterior_gamma_grid(
    x, ybar, nrep, within_sse, n_total, log_gamma_grid
)

idx_hat = np.argmin(np.abs(gamma_grid - gamma_hat))
sigma_samps = posterior_sigma_given_gamma(S_list[idx_hat], n_total, rng, size=1000)
sigma_hat = np.mean(sigma_samps)

x_grid = np.linspace(0, 63, 300)
f_post_mean = posterior_mean_f_mixture(
    x, ybar, x_grid, gamma_grid, gamma_weights, A_list, beta0_list
)

print("full sample size =", len(df))
print("unique experience values =", len(x))
print("beta0 hat =", beta0_list[idx_hat])
print("gamma hat =", gamma_hat)
print("sigma hat =", sigma_hat)

plt.figure(figsize=(10, 5))
plt.scatter(df["Exper"], df["log_weekly_earnings"], s=6, alpha=0.15, label="Data")
plt.plot(x_grid, f_post_mean, linewidth=2, label="Posterior mean")
plt.title("Posterior mean estimate on full data")
plt.xlabel("Experience")
plt.ylabel("Log weekly earnings")
plt.legend()
plt.tight_layout()
plt.show()
