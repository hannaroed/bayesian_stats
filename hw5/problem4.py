import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import pymc as pm
import arviz as az

# data
years = np.arange(1976, 1986)
y = np.array([24, 25, 31, 31, 22, 21, 26, 20, 16, 22])

X = np.column_stack([np.ones_like(years), years])
param_names = ["alpha", "beta"]

n, p = X.shape

# large box prior: alpha, beta ~ Uniform(-M, M) independently
M = 400.0

### 4a) ###

# Laplace posterior normal approximation
laplace_model = sm.GLM(y, X, family=sm.families.Poisson())
laplace_res = laplace_model.fit()

mode = np.asarray(laplace_res.params)
laplace_cov = np.asarray(laplace_res.cov_params())
laplace_sd = np.sqrt(np.diag(laplace_cov))

if np.any(np.abs(mode) >= M):
    raise ValueError("Posterior mode is not inside the box prior. Increase M.")

print("Laplace posterior normal approximation:")
for name, est, se in zip(param_names, mode, laplace_sd):
    print(f"{name:>10s}: mode = {est: .6f}, SD = {se: .6f}")

print("\nApproximate posterior covariance matrix:")
print(laplace_cov)

### 4b) ###

# Log-posterior under Poisson log-linear model with box prior
def logpost(theta):
    theta = np.asarray(theta)
    alpha, beta = theta

    if np.any(theta < -M) or np.any(theta > M):
        return -np.inf

    eta = alpha + beta * years

    if np.any(eta > 700):
        return -np.inf

    return np.sum(y * eta - np.exp(eta))

# metropolis-Hastings sampler
prop_cov = (2.4**2 / p) * laplace_cov
prop_cov = 0.5 * (prop_cov + prop_cov.T) + 1e-10 * np.eye(p)

def metropolis_hastings(n_iter=120000, burn=20000, start=None, seed=123):
    rng = np.random.default_rng(seed)

    if start is None:
        start = mode.copy()

    theta = start.copy()
    logp_theta = logpost(theta)

    samples = np.zeros((n_iter, p))
    accept = 0

    for t in range(n_iter):
        proposal = rng.multivariate_normal(theta, prop_cov)
        logp_prop = logpost(proposal)

        # symmetric proposal, so Hastings ratio cancels
        if np.log(rng.random()) < (logp_prop - logp_theta):
            theta = proposal
            logp_theta = logp_prop
            accept += 1

        samples[t] = theta

    return samples[burn:], accept / n_iter


samples_mh, acc_rate = metropolis_hastings()

mh_mean = samples_mh.mean(axis=0)
mh_sd = samples_mh.std(axis=0, ddof=1)

print("\nMH acceptance rate =", acc_rate)
print("\nLaplace vs MH comparison:")
for name, est, se, post_mean, post_sd in zip(param_names, mode, laplace_sd, mh_mean, mh_sd):
    print(
        f"{name:>10s}: "
        f"Laplace mean = {est: .6f}, "
        f"Laplace SD = {se: .6f}, "
        f"MH mean = {post_mean: .6f}, "
        f"MH SD = {post_sd: .6f}"
    )

# trace plots for MH
fig, axes = plt.subplots(p, 1, figsize=(10, 4.5), sharex=True)

for j in range(p):
    axes[j].plot(samples_mh[:5000, j], lw=0.5)
    axes[j].set_ylabel(param_names[j])

axes[-1].set_xlabel("Iteration")
fig.suptitle("MH trace plots (first 5000 post-burn-in draws)", y=0.995)
plt.tight_layout()
plt.show()

# histograms for MH posterior samples
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for j in range(p):
    axes[j].hist(samples_mh[:, j], bins=50, density=True)
    xgrid = np.linspace(samples_mh[:, j].min(), samples_mh[:, j].max(), 400)
    axes[j].plot(xgrid, norm.pdf(xgrid, loc=mode[j], scale=laplace_sd[j]))
    axes[j].axvline(mode[j], linestyle="--")
    axes[j].set_title(param_names[j])

plt.tight_layout()
plt.show()

### 4c) ###

# PyMC Poisson log-linear model
with pm.Model() as model:
    alpha = pm.Uniform("alpha", lower=-M, upper=M)
    beta = pm.Uniform("beta", lower=-M, upper=M)

    mu = pm.math.exp(alpha + beta * years)

    pm.Poisson("y_obs", mu=mu, observed=y)

    idata = pm.sample(
        draws=3000,
        tune=2000,
        chains=4,
        target_accept=0.95,
        random_seed=123,
        return_inferencedata=True,
        init="adapt_diag",
        nuts={"max_treedepth": 15}
    )

print("\nPyMC summary:")
print(az.summary(idata, var_names=["alpha", "beta"]))

az.plot_trace(idata, var_names=["alpha", "beta"])
plt.tight_layout()
plt.show()

# extract PyMC posterior draws and compare with MH
alpha_draws = idata.posterior["alpha"].values.reshape(-1)
beta_draws = idata.posterior["beta"].values.reshape(-1)
posterior_draws = np.column_stack([alpha_draws, beta_draws])

pymc_mean = posterior_draws.mean(axis=0)
pymc_sd = posterior_draws.std(axis=0, ddof=1)

print("\nPyMC posterior summaries:")
for name, post_mean, post_sd in zip(param_names, pymc_mean, pymc_sd):
    print(f"{name:>10s}: mean = {post_mean: .6f}, SD = {post_sd: .6f}")

print("\nMH vs PyMC comparison:")
for name, mhm, mhs, pym, pys in zip(param_names, mh_mean, mh_sd, pymc_mean, pymc_sd):
    print(
        f"{name:>10s}: "
        f"MH mean = {mhm: .6f}, MH SD = {mhs: .6f}, "
        f"PyMC mean = {pym: .6f}, PyMC SD = {pys: .6f}"
    )

### 4d) ###

# posterior predictive draws for 1986
t_pred = 1986
rng = np.random.default_rng(2026)

laplace_draws = rng.multivariate_normal(mode, laplace_cov, size=20000)
lam_laplace = np.exp(np.clip(laplace_draws[:, 0] + laplace_draws[:, 1] * t_pred, -700, 700))
pp_laplace = rng.poisson(lam_laplace)

lam_mh = np.exp(np.clip(samples_mh[:, 0] + samples_mh[:, 1] * t_pred, -700, 700))
pp_mh = rng.poisson(lam_mh)

lam_pymc = np.exp(np.clip(posterior_draws[:, 0] + posterior_draws[:, 1] * t_pred, -700, 700))
pp_pymc = rng.poisson(lam_pymc)

ci_laplace = np.quantile(pp_laplace, [0.025, 0.975])
ci_mh = np.quantile(pp_mh, [0.025, 0.975])
ci_pymc = np.quantile(pp_pymc, [0.025, 0.975])

print("\n95% posterior predictive credible intervals for 1986:")
print("Laplace:", ci_laplace)
print("MH     :", ci_mh)
print("PyMC   :", ci_pymc)

# histograms for posterior predictive draws
all_pp = np.concatenate([pp_laplace, pp_mh, pp_pymc])
bins = np.arange(all_pp.min() - 0.5, all_pp.max() + 1.5, 1)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].hist(pp_laplace, bins=bins, density=True)
axes[0].set_title("Posterior predictive for 1986: Laplace")

axes[1].hist(pp_mh, bins=bins, density=True)
axes[1].set_title("Posterior predictive for 1986: MH")

axes[2].hist(pp_pymc, bins=bins, density=True)
axes[2].set_title("Posterior predictive for 1986: PyMC")
axes[2].set_xlabel("Number of fatal accidents in 1986")

plt.tight_layout()
plt.show()
