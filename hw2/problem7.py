import logging
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import gammaln

logging.getLogger("pymc").setLevel(logging.ERROR)

url = "https://raw.githubusercontent.com/berkeley-stat238/spring-2026/main/KidneyCancerClean.csv"
df = pd.read_csv(url, skiprows=4)

x = df["dc"].to_numpy(dtype=float)
n = df["pop"].to_numpy(dtype=float)
county_col = "Location"

def poisson_gamma_log_marginal(alpha, beta, x_arr, n_arr):
    out = (
        gammaln(alpha + x_arr)
        - gammaln(alpha)
        - gammaln(x_arr + 1.0)
        + x_arr * (np.log(n_arr) - np.log(beta + n_arr))
        + alpha * (np.log(beta) - np.log(beta + n_arr))
    )
    return float(np.sum(out))

### 7(b) ###

log_alpha_grid = np.linspace(np.log(0.1), np.log(100.0), 180)
log_beta_grid = np.linspace(np.log(100.0), np.log(3_000_000.0), 220)
alpha_grid = np.exp(log_alpha_grid)
beta_grid = np.exp(log_beta_grid)

best_ll = -np.inf
alpha_hat = np.nan
beta_hat = np.nan

for a in alpha_grid:
    for b in beta_grid:
        ll = poisson_gamma_log_marginal(a, b, x, n)
        if ll > best_ll:
            best_ll = ll
            alpha_hat = a
            beta_hat = b

print("alpha_hat =", float(alpha_hat))
print("beta_hat  =", float(beta_hat))

### 7(c) ###

theta_hat = (alpha_hat + x) / (beta_hat + n)
df["theta_post_mean_pg"] = theta_hat

top100_pg = df.sort_values("theta_post_mean_pg", ascending=False).head(100)[county_col].astype(str).tolist()

a_class = 11.781889938777498
b_class = 120049.39668603126

theta_class = (a_class + x) / (a_class + b_class + n)
df["theta_post_mean_bb"] = theta_class

top100_bb = df.iloc[np.argsort(-theta_class)[:100]][county_col].astype(str).tolist()

top100_pg_set = set(top100_pg)
top100_bb_set = set(top100_bb)
overlap = top100_pg_set.intersection(top100_bb_set)
only_pg = top100_pg_set - top100_bb_set
only_bb = top100_bb_set - top100_pg_set

print("Shared counties in top 100:", len(overlap))
print("Poisson-Gamma only (top 100):", len(only_pg))
print("Class Beta-Binomial only (top 100):", len(only_bb))

print("\nTop 20 counties by Poisson-Gamma:")
for county in top100_pg[:20]:
    print(" ", county)

print("\nTop 20 counties by class Beta-Binomial:")
for county in top100_bb[:20]:
    print(" ", county)

### 7(d) ###

logw = np.zeros((len(alpha_grid), len(beta_grid)))

for i, a in enumerate(alpha_grid):
    for j, b in enumerate(beta_grid):
        ll = poisson_gamma_log_marginal(a, b, x, n)
        logw[i, j] = ll

m = np.max(logw)
w = np.exp(logw - m)
w = w / np.sum(w)

theta_full_bayes = np.zeros(len(x), dtype=np.float64)
for i, a in enumerate(alpha_grid):
    for j, b in enumerate(beta_grid):
        theta_full_bayes += w[i, j] * ((a + x) / (b + n))

df["theta_full_bayes_pg"] = theta_full_bayes
top10_fb = df.sort_values("theta_full_bayes_pg", ascending=False).head(10)
print("min theta_full_bayes =", float(theta_full_bayes.min()))
print("max theta_full_bayes =", float(theta_full_bayes.max()))
print("mean theta_full_bayes =", float(theta_full_bayes.mean()))
print("Top 10 counties (full Bayes):")
print(top10_fb[[county_col, "theta_full_bayes_pg"]].to_string(index=False))


### 7(e) ###

with pm.Model() as model:
    log_alpha = pm.Uniform("log_alpha", lower=np.log(alpha_grid.min()), upper=np.log(alpha_grid.max()))
    log_beta = pm.Uniform("log_beta", lower=np.log(beta_grid.min()), upper=np.log(beta_grid.max()))

    alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
    beta = pm.Deterministic("beta", pm.math.exp(log_beta))

    p = beta / (beta + n)
    pm.NegativeBinomial("Xobs", n=alpha, p=p, observed=x)

    idata = pm.sample(
        draws=1500,
        tune=1500,
        chains=2,
        target_accept=0.9,
        progressbar=False,
        random_seed=238,
    )

alpha_draws = idata.posterior["alpha"].values.reshape(-1, 1)
beta_draws = idata.posterior["beta"].values.reshape(-1, 1)

theta_pymc_mean = np.mean((alpha_draws + x[None, :]) / (beta_draws + n[None, :]), axis=0)

mad = np.mean(np.abs(theta_pymc_mean - theta_full_bayes))
corr = np.corrcoef(theta_pymc_mean, theta_full_bayes)[0, 1]

print("Mean abs diff =", float(mad))
print("Correlation   =", float(corr))

top100_pymc_idx = np.argsort(-theta_pymc_mean)[:100]
top100_grid_idx = np.argsort(-theta_full_bayes)[:100]
overlap_idx = len(set(top100_pymc_idx).intersection(set(top100_grid_idx)))

print("Top-100 overlap count (PyMC vs grid) =", overlap_idx)

if county_col in df.columns:
    top100_pymc_counties = set(df.iloc[top100_pymc_idx][county_col].astype(str))
    top100_grid_counties = set(df.iloc[top100_grid_idx][county_col].astype(str))
    overlap_counties = len(top100_pymc_counties.intersection(top100_grid_counties))
    print("Top-100 county-name overlap (PyMC vs grid) =", overlap_counties)
