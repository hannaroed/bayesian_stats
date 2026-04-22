import numpy as np
import matplotlib.pyplot as plt

x = np.array([44.0, 43.0, 47.5, 46.5, 45.0])

### 1a) ###

grid = np.linspace(0, 100, 20001)

log_post = np.full(len(grid), -np.inf)
for i in range(len(grid)):
    theta = grid[i]
    log_post[i] = np.sum(-np.log(np.pi * (1 + (x - theta)**2)))

m = np.max(log_post)
post_unnorm = np.exp(log_post - m)
post = post_unnorm / np.trapezoid(post_unnorm, grid)

post_mode = grid[np.argmax(post)]
post_mean = np.trapezoid(grid * post, grid)
post_var = np.trapezoid(((grid - post_mean)**2) * post, grid)
post_sd = np.sqrt(post_var)

dx = grid[1] - grid[0]
cdf = np.cumsum(post) * dx
cdf = cdf / cdf[-1]

q025 = grid[np.searchsorted(cdf, 0.025)]
q50 = grid[np.searchsorted(cdf, 0.50)]
q975 = grid[np.searchsorted(cdf, 0.975)]

print("Numerical posterior on a fine grid")
print("Posterior mode =", round(post_mode, 4))
print("Posterior mean =", round(post_mean, 4))
print("Posterior sd =", round(post_sd, 4))
print("2.5% quantile =", round(q025, 4))
print("50% quantile =", round(q50, 4))
print("97.5% quantile =", round(q975, 4))

plt.figure(figsize=(8, 5))
plt.plot(grid, post)
plt.xlabel("theta")
plt.ylabel("posterior density")
plt.title("Numerical posterior of theta")
plt.show()

### 1b) ###

rng = np.random.default_rng(123)

n_iter = 60000
burn = 10000
proposal_sd = 1.0

mh_chain = np.zeros(n_iter)
theta_current = 45.0

if 0 <= theta_current <= 100:
    log_post_current = np.sum(-np.log(np.pi * (1 + (x - theta_current)**2)))
else:
    log_post_current = -np.inf

accept = 0

for t in range(n_iter):
    theta_proposed = rng.normal(theta_current, proposal_sd)

    if 0 <= theta_proposed <= 100:
        log_post_proposed = np.sum(-np.log(np.pi * (1 + (x - theta_proposed)**2)))
    else:
        log_post_proposed = -np.inf

    log_alpha = log_post_proposed - log_post_current

    if np.log(rng.uniform()) < log_alpha:
        theta_current = theta_proposed
        log_post_current = log_post_proposed
        accept += 1

    mh_chain[t] = theta_current

mh_samples = mh_chain[burn:]

mh_mean = np.mean(mh_samples)
mh_sd = np.std(mh_samples, ddof=1)
mh_q025 = np.quantile(mh_samples, 0.025)
mh_q50 = np.quantile(mh_samples, 0.50)
mh_q975 = np.quantile(mh_samples, 0.975)
accept_rate = accept / n_iter

print("\nMetropolis-Hastings")
print("Proposal used: theta* | theta ~ N(theta, 1.0^2)")
print("Iterations =", n_iter)
print("Burn-in =", burn)
print("Kept samples =", len(mh_samples))
print("Acceptance rate =", round(accept_rate, 4))
print("Posterior mean =", round(mh_mean, 4))
print("Posterior sd =", round(mh_sd, 4))
print("2.5% quantile =", round(mh_q025, 4))
print("50% quantile =", round(mh_q50, 4))
print("97.5% quantile =", round(mh_q975, 4))

plt.figure(figsize=(8, 5))
plt.plot(mh_chain[:5000])
plt.xlabel("iteration")
plt.ylabel("theta")
plt.title("MH trace plot (first 5000 iterations)")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(mh_samples, bins=50, density=True, alpha=0.6, label="MH samples")
plt.plot(grid, post, linewidth=2, label="Numerical posterior")
plt.xlabel("theta")
plt.ylabel("density")
plt.title("MH histogram vs numerical posterior")
plt.legend()
plt.show()

### 1c) ###

import pymc as pm
import arviz as az

with pm.Model() as model:
    theta = pm.Uniform("theta", lower=0, upper=100)
    obs = pm.Cauchy("obs", alpha=theta, beta=1, observed=x)

    idata = pm.sample(
        draws=6000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=123,
        return_inferencedata=True
    )

pymc_samples = idata.posterior["theta"].values.flatten()

pymc_mean = np.mean(pymc_samples)
pymc_sd = np.std(pymc_samples, ddof=1)
pymc_q025 = np.quantile(pymc_samples, 0.025)
pymc_q50 = np.quantile(pymc_samples, 0.50)
pymc_q975 = np.quantile(pymc_samples, 0.975)

print("\nPyMC")
print(az.summary(idata, var_names=["theta"]))
print("Posterior mean =", round(pymc_mean, 4))
print("Posterior sd =", round(pymc_sd, 4))
print("2.5% quantile =", round(pymc_q025, 4))
print("50% quantile =", round(pymc_q50, 4))
print("97.5% quantile =", round(pymc_q975, 4))

plt.figure(figsize=(8, 5))
plt.hist(pymc_samples, bins=50, density=True, alpha=0.6, label="PyMC samples")
plt.plot(grid, post, linewidth=2, label="Numerical posterior")
plt.xlabel("theta")
plt.ylabel("density")
plt.title("PyMC histogram vs numerical posterior")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(mh_samples, bins=50, density=True, alpha=0.45, label="MH samples")
plt.hist(pymc_samples, bins=50, density=True, alpha=0.45, label="PyMC samples")
plt.plot(grid, post, linewidth=2, label="Numerical posterior")
plt.xlabel("theta")
plt.ylabel("density")
plt.title("MH vs PyMC vs numerical posterior")
plt.legend()
plt.show()

print("\nComparison")
print("Grid mean and sd:", round(post_mean, 4), round(post_sd, 4))
print("MH mean and sd:", round(mh_mean, 4), round(mh_sd, 4))
print("PyMC mean and sd:", round(pymc_mean, 4), round(pymc_sd, 4))