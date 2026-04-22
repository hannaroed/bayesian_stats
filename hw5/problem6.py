import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

groups = {
    "A": np.array([62, 60, 63, 59], dtype=float),
    "B": np.array([63, 67, 71, 64, 65, 66], dtype=float),
    "C": np.array([68, 66, 71, 67, 68, 68], dtype=float),
    "D": np.array([56, 62, 60, 61, 63, 64, 63, 59], dtype=float),
}

group_names = list(groups.keys())
y_list = [groups[g] for g in group_names]
J = len(y_list)
n_j = np.array([len(y) for y in y_list])
ybar_j = np.array([y.mean() for y in y_list])
all_y = np.concatenate(y_list)
N_total = len(all_y)

print("Group sizes:", dict(zip(group_names, n_j)))
print("Group means:", dict(zip(group_names, ybar_j)))
print("Overall mean:", all_y.mean())

### 6b) ###

# Gamma(a,b) is used in shape-rate form, so NumPy needs scale = 1 / rate

def gibbs_hier_normal(n_keep=30000, burn=5000, seed=238):
    rng = np.random.default_rng(seed)

    # Initialize
    theta = ybar_j.copy()
    mu = all_y.mean()
    sigma2 = np.var(all_y, ddof=1)
    tau2 = np.var(theta, ddof=1)
    if tau2 <= 0:
        tau2 = 1.0

    samples = np.zeros((n_keep, J + 3))
    # columns: theta_A, theta_B, theta_C, theta_D, mu, tau, sigma

    a0 = 0.001
    b0 = 0.001

    total_iters = burn + n_keep

    for t in range(total_iters):
        lambda_sigma = 1.0 / sigma2
        lambda_tau = 1.0 / tau2

        # Update theta_j
        for j in range(J):
            V_j = 1.0 / (n_j[j] * lambda_sigma + lambda_tau)
            m_j = V_j * (lambda_sigma * np.sum(y_list[j]) + lambda_tau * mu)
            theta[j] = rng.normal(m_j, np.sqrt(V_j))

        # Update mu
        V_mu = 1.0 / (J * lambda_tau + 1.0 / (100.0**2))
        m_mu = V_mu * (lambda_tau * np.sum(theta))
        mu = rng.normal(m_mu, np.sqrt(V_mu))

        # Update lambda_tau = 1 / tau^2
        shape_tau = a0 + J / 2.0
        rate_tau = b0 + 0.5 * np.sum((theta - mu) ** 2)
        lambda_tau = rng.gamma(shape=shape_tau, scale=1.0 / rate_tau)
        tau2 = 1.0 / lambda_tau

        # Update lambda_sigma = 1 / sigma^2
        ss_within = 0.0
        for j in range(J):
            ss_within += np.sum((y_list[j] - theta[j]) ** 2)

        shape_sigma = a0 + N_total / 2.0
        rate_sigma = b0 + 0.5 * ss_within
        lambda_sigma = rng.gamma(shape=shape_sigma, scale=1.0 / rate_sigma)
        sigma2 = 1.0 / lambda_sigma

        # Store after burn-in
        if t >= burn:
            idx = t - burn
            samples[idx, :J] = theta
            samples[idx, J] = mu
            samples[idx, J + 1] = np.sqrt(tau2)
            samples[idx, J + 2] = np.sqrt(sigma2)

    return samples

gibbs_samples = gibbs_hier_normal(n_keep=30000, burn=5000, seed=238)

param_names = [f"theta_{g}" for g in group_names] + ["mu", "tau", "sigma"]

# Posterior summary
gibbs_summary = pd.DataFrame({
    "mean": gibbs_samples.mean(axis=0),
    "sd": gibbs_samples.std(axis=0, ddof=1),
    "q2.5": np.quantile(gibbs_samples, 0.025, axis=0),
    "median": np.quantile(gibbs_samples, 0.50, axis=0),
    "q97.5": np.quantile(gibbs_samples, 0.975, axis=0),
}, index=param_names)

print("\nGibbs posterior summary:")
print(gibbs_summary.round(4))

# Trace plots
fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 14), sharex=True)
for i, name in enumerate(param_names):
    axes[i].plot(gibbs_samples[:, i], lw=0.5)
    axes[i].set_ylabel(name)
axes[-1].set_xlabel("Iteration")
fig.suptitle("Gibbs sampler trace plots", y=0.995)
plt.tight_layout()
plt.show()

# Histograms
fig, axes = plt.subplots(len(param_names), 1, figsize=(8, 16))
for i, name in enumerate(param_names):
    axes[i].hist(gibbs_samples[:, i], bins=40, density=True)
    axes[i].set_title(f"Posterior histogram: {name}")
plt.tight_layout()
plt.show()

### 6c) ###

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0.0, sigma=100.0)

    lambda_tau = pm.Gamma("lambda_tau", alpha=0.001, beta=0.001)
    lambda_sigma = pm.Gamma("lambda_sigma", alpha=0.001, beta=0.001)

    tau = pm.Deterministic("tau", 1 / pm.math.sqrt(lambda_tau))
    sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(lambda_sigma))

    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)

    for j in range(J):
        pm.Normal(f"y_obs_{j}", mu=theta[j], sigma=sigma, observed=y_list[j])

    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=238,
        return_inferencedata=True,
    )

print("\nPyMC posterior summary:")
pymc_summary = az.summary(idata, var_names=["theta", "mu", "tau", "sigma"], round_to=4)
print(pymc_summary)

az.plot_trace(idata, var_names=["theta", "mu", "tau", "sigma"])
plt.tight_layout()
plt.show()

# Build comparison table
pymc_means = [
    pymc_summary.loc["theta[0]", "mean"],
    pymc_summary.loc["theta[1]", "mean"],
    pymc_summary.loc["theta[2]", "mean"],
    pymc_summary.loc["theta[3]", "mean"],
    pymc_summary.loc["mu", "mean"],
    pymc_summary.loc["tau", "mean"],
    pymc_summary.loc["sigma", "mean"],
]

pymc_sds = [
    pymc_summary.loc["theta[0]", "sd"],
    pymc_summary.loc["theta[1]", "sd"],
    pymc_summary.loc["theta[2]", "sd"],
    pymc_summary.loc["theta[3]", "sd"],
    pymc_summary.loc["mu", "sd"],
    pymc_summary.loc["tau", "sd"],
    pymc_summary.loc["sigma", "sd"],
]

comparison = pd.DataFrame({
    "Gibbs mean": gibbs_summary["mean"].values,
    "Gibbs sd": gibbs_summary["sd"].values,
    "PyMC mean": pymc_means,
    "PyMC sd": pymc_sds,
}, index=param_names)

print("\nComparison of Gibbs and PyMC:")
print(comparison.round(4))

