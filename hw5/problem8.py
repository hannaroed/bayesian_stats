import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import norm

# data
y1 = np.array([
    52.11, 57.65, 66.44, 44.68, 40.57, 35.04, 50.71, 66.17,
    39.43, 46.17, 58.76, 47.97, 39.18, 64.63, 69.38, 32.38,
    29.98, 59.32, 43.04, 57.83, 46.07, 47.74, 48.66, 40.80,
    66.32, 53.70, 52.42, 71.38, 59.66, 47.52, 39.51
])

y2 = np.array([
    57.57, 42.40, 41.41, 55.22, 43.90, 53.04, 49.00, 62.45,
    53.78, 49.08, 40.25, 43.08, 52.43, 21.73, 53.68, 41.45,
    45.47, 34.06, 33.45, 60.78, 35.92, 52.40
])

n1, n2 = len(y1), len(y2)
n = n1 + n2

# prior hyperparameters
mu0, Vmu0 = 50.0, 625.0
delta0, Vdelta0 = 0.0, 625.0
a0, b0 = 0.5, 50.0 # Gamma(shape, rate)

### 8a) ###
rng = np.random.default_rng(42)

N = 80000
burn = 10000

mu = np.mean(np.r_[y1, y2])
delta = (y1.mean() - y2.mean()) / 2
tau = 1 / np.var(np.r_[y1, y2], ddof=1)

mu_draws = np.empty(N)
delta_draws = np.empty(N)
sigma_draws = np.empty(N)

for t in range(N):
    # mu | delta, tau, y
    prec_mu = tau * n + 1 / Vmu0
    var_mu = 1 / prec_mu
    mean_mu = var_mu * (
        tau * (np.sum(y1 - delta) + np.sum(y2 + delta)) + mu0 / Vmu0
    )
    mu = rng.normal(mean_mu, np.sqrt(var_mu))

    # delta | mu, tau, y
    prec_delta = tau * n + 1 / Vdelta0
    var_delta = 1 / prec_delta
    mean_delta = var_delta * (
        tau * (np.sum(y1 - mu) + np.sum(mu - y2)) + delta0 / Vdelta0
    )
    delta = rng.normal(mean_delta, np.sqrt(var_delta))

    # tau | mu, delta, y
    sse = np.sum((y1 - mu - delta) ** 2) + np.sum((y2 - mu + delta) ** 2)
    shape_post = a0 + n / 2
    rate_post = b0 + 0.5 * sse
    tau = rng.gamma(shape=shape_post, scale=1 / rate_post)

    mu_draws[t] = mu
    delta_draws[t] = delta
    sigma_draws[t] = 1 / np.sqrt(tau)

mu_g = mu_draws[burn:]
delta_g = delta_draws[burn:]
sigma_g = sigma_draws[burn:]

print("Gibbs posterior summaries")
print(f"mu:    mean = {mu_g.mean():.6f}, sd = {mu_g.std(ddof=1):.6f}")
print(f"delta: mean = {delta_g.mean():.6f}, sd = {delta_g.std(ddof=1):.6f}")
print(f"sigma: mean = {sigma_g.mean():.6f}, sd = {sigma_g.std(ddof=1):.6f}")

### 8b) ###
with pm.Model() as model:
    mu_pm = pm.Normal("mu", mu=50, sigma=25)
    delta_pm = pm.Normal("delta", mu=0, sigma=25)
    tau_pm = pm.Gamma("tau", alpha=0.5, beta=50) # beta = rate
    sigma_pm = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau_pm))

    y1_obs = pm.Normal("y1_obs", mu=mu_pm + delta_pm, sigma=sigma_pm, observed=y1)
    y2_obs = pm.Normal("y2_obs", mu=mu_pm - delta_pm, sigma=sigma_pm, observed=y2)

    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=1,
        random_seed=42,
        target_accept=0.9,
        progressbar=False
    )

summary = az.summary(idata, var_names=["mu", "delta", "sigma"])
print("\nPyMC posterior summaries")
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])

mu_p = idata.posterior["mu"].values.reshape(-1)
delta_p = idata.posterior["delta"].values.reshape(-1)
sigma_p = idata.posterior["sigma"].values.reshape(-1)

### 8c) ###

p_delta_gt_0_gibbs = np.mean(delta_g > 0)
p_delta_gt_0_pymc = np.mean(delta_p > 0)

print("\nPosterior probability delta > 0")
print(f"Gibbs: {p_delta_gt_0_gibbs:.6f}")
print(f"PyMC:  {p_delta_gt_0_pymc:.6f}")

### 8d) ###

p_school1_gt_school2_gibbs = np.mean(norm.cdf(np.sqrt(2) * delta_g / sigma_g))
p_school1_gt_school2_pymc = np.mean(norm.cdf(np.sqrt(2) * delta_p / sigma_p))

print("\nPosterior probability a random School One student outscores a random School Two student")
print(f"Gibbs: {p_school1_gt_school2_gibbs:.6f}")
print(f"PyMC:  {p_school1_gt_school2_pymc:.6f}")
