import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import pymc as pm
import arviz as az

# data
df = pd.read_csv("hw5/data/frogs.csv")

y = df["pres.abs"].to_numpy()
X_df = sm.add_constant(df.drop(columns=["pres.abs"]), has_constant="add")
X = X_df.to_numpy()
param_names = X_df.columns.tolist()

n, p = X.shape

# large box prior: beta_j ~ Uniform(-C, C) independently
C = 200.0
coords = {"coef": param_names}

### 3a) ###

# frequentist probit fit for comparison
freq_model = sm.Probit(y, X)
freq_res = freq_model.fit(disp=False)

mle = np.asarray(freq_res.params)
freq_se = np.asarray(freq_res.bse)
freq_cov = np.asarray(freq_res.cov_params())

print("Frequentist probit estimates:")
for name, est, se in zip(param_names, mle, freq_se):
    print(f"{name:>10s}: estimate = {est: .6f}, SE = {se: .6f}")

# Log-posterior under probit regression with box prior
def logpost(beta):
    beta = np.asarray(beta)

    if np.any(beta < -C) or np.any(beta > C):
        return -np.inf

    eta = X @ beta

    # Stable log-likelihood for probit model:
    # log Phi(eta) when y=1, log Phi(-eta) when y=0
    return np.sum(y * norm.logcdf(eta) + (1 - y) * norm.logcdf(-eta))

# metropolis-Hastings sampler
prop_cov = (2.4**2 / p) * freq_cov

def metropolis_hastings(n_iter=120000, burn=20000, start=None, seed=123):
    rng = np.random.default_rng(seed)

    if start is None:
        start = mle.copy()

    beta = start.copy()
    logp_beta = logpost(beta)

    samples = np.zeros((n_iter, p))
    accept = 0

    for t in range(n_iter):
        proposal = rng.multivariate_normal(beta, prop_cov)
        logp_prop = logpost(proposal)

        # symmetric proposal, so Hastings ratio cancels
        if np.log(rng.random()) < (logp_prop - logp_beta):
            beta = proposal
            logp_beta = logp_prop
            accept += 1

        samples[t] = beta

    return samples[burn:], accept / n_iter


samples_mh, acc_rate = metropolis_hastings()

mh_mean = samples_mh.mean(axis=0)
mh_sd = samples_mh.std(axis=0, ddof=1)

print("\nMH acceptance rate =", acc_rate)
print("\nComparison table:")
for name, est, se, post_mean, post_sd in zip(param_names, mle, freq_se, mh_mean, mh_sd):
    print(
        f"{name:>10s}: "
        f"freq est = {est: .6f}, "
        f"freq SE = {se: .6f}, "
        f"MH mean = {post_mean: .6f}, "
        f"MH SD = {post_sd: .6f}"
    )

# trace plots for MH
fig, axes = plt.subplots(p, 1, figsize=(10, 2.2 * p), sharex=True)

for j in range(p):
    axes[j].plot(samples_mh[:3000, j], lw=0.5)
    axes[j].set_ylabel(param_names[j])

axes[-1].set_xlabel("Iteration")
fig.suptitle("MH trace plots (first 3000 post-burn-in draws)", y=0.995)
plt.tight_layout()
plt.show()

# histograms for MH posterior samples
fig, axes = plt.subplots(5, 2, figsize=(12, 14))
axes = axes.ravel()

for j in range(p):
    axes[j].hist(samples_mh[:, j], bins=50, density=True)
    axes[j].axvline(mle[j], linestyle="--")
    axes[j].set_title(param_names[j])

plt.tight_layout()
plt.show()

### 3b) ###

# PyMC probit model
with pm.Model(coords=coords) as model:
    beta = pm.Uniform("beta", lower=-C, upper=C, dims="coef")

    eta = pm.math.dot(X, beta)
    prob = 0.5 * (1 + pm.math.erf(eta / pm.math.sqrt(2.0)))

    pm.Bernoulli("y_obs", p=prob, observed=y)

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
print(az.summary(idata, var_names=["beta"]))

az.plot_trace(idata, var_names=["beta"])
plt.tight_layout()
plt.show()

# extract PyMC posterior draws and compare with MH
posterior_draws = (
    idata.posterior["beta"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "coef")
    .values
)
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
