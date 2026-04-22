import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
import pymc as pm
import arviz as az

# data
x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])

# prior bounds
alpha_bounds = (-20, 20)
beta_bounds = (-20, 40)
ld50_quantiles = [0.025, 0.5, 0.975]

def logpost(theta):
    alpha, beta = theta

    # uniform prior on the rectangle
    if not (alpha_bounds[0] <= alpha <= alpha_bounds[1] and
            beta_bounds[0] <= beta <= beta_bounds[1]):
        return -np.inf

    z = alpha + beta * x

    # stable computation of log(1 + exp(z))
    return np.sum(y * z - n * np.logaddexp(0.0, z))

def summarize_ld50(draws):
    keep = draws[:, 1] > 0
    ld50 = -draws[keep, 0] / draws[keep, 1]
    return keep.mean(), ld50


def plot_ld50_hist(ld50, title):
    hist_range = np.quantile(ld50, [0.001, 0.999])
    plt.figure(figsize=(7, 4.5))
    plt.hist(ld50, bins=60, density=True, range=hist_range)
    plt.xlim(hist_range)
    plt.xlabel(r"$-\alpha/\beta$")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.show()

### 2a) ###

def neglogpost(theta):
    val = logpost(theta)
    return -val if np.isfinite(val) else 1e100

# using bounded optimization since the prior is uniform on a rectangle
res = minimize(
    neglogpost,
    x0=np.array([0.0, 10.0]),
    method="L-BFGS-B",
    bounds=[alpha_bounds, beta_bounds]
)

if not res.success:
    raise RuntimeError(f"Optimization failed: {res.message}")

mode = res.x

def neg_hessian(theta):
    alpha, beta = theta
    z = alpha + beta * x
    p = expit(z)
    w = n * p * (1 - p)

    H = np.array([
        [np.sum(w),       np.sum(w * x)],
        [np.sum(w * x),   np.sum(w * x * x)]
    ])
    return H

H = neg_hessian(mode)
Sigma = np.linalg.inv(H)

print("posterior mode =", mode)
print("covariance matrix =")
print(Sigma)

mu_beta = mode[1]
sd_beta = np.sqrt(Sigma[1, 1])
prob_beta_gt_0_laplace = 1 - norm.cdf((0 - mu_beta) / sd_beta)
print("P(beta > 0) under Laplace approximation =", prob_beta_gt_0_laplace)

rng = np.random.default_rng(123)
norm_draws = rng.multivariate_normal(mean=mode, cov=Sigma, size=200000)
_, ld50_norm = summarize_ld50(norm_draws)

print("Conditional LD50 quantiles under Laplace approximation:")
print(np.quantile(ld50_norm, ld50_quantiles))

plot_ld50_hist(
    ld50_norm,
    r"Laplace approximation: posterior of $-\alpha/\beta$ given $\beta>0$"
)

### 2b) ###

# random-walk proposal covariance
prop_cov = (2.4**2 / 2.0) * Sigma

def metropolis_hastings(n_iter=120000, burn=20000, start=None, seed=123):
    rng = np.random.default_rng(seed)

    if start is None:
        start = mode.copy()

    samples = np.zeros((n_iter, 2))
    theta = start.copy()
    logp_theta = logpost(theta)
    accept = 0

    for t in range(n_iter):
        proposal = rng.multivariate_normal(theta, prop_cov)
        logp_prop = logpost(proposal)

        if np.log(rng.random()) < (logp_prop - logp_theta):
            theta = proposal
            logp_theta = logp_prop
            accept += 1

        samples[t] = theta

    return samples[burn:], accept / n_iter


samples_mh, acc_rate = metropolis_hastings()

print("acceptance rate =", acc_rate)
print("posterior mean =", samples_mh.mean(axis=0))
print("posterior sd =", samples_mh.std(axis=0, ddof=1))

prob_beta_gt_0_mh, ld50_mh = summarize_ld50(samples_mh)

print("P(beta > 0) from MH =", prob_beta_gt_0_mh)
print("LD50 quantiles from MH =", np.quantile(ld50_mh, ld50_quantiles))

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].plot(samples_mh[:3000, 0], lw=0.5)
ax[0].set_ylabel(r"$\alpha$")
ax[0].set_title("MH trace plots (first 3000 post-burn-in draws)")

ax[1].plot(samples_mh[:3000, 1], lw=0.5)
ax[1].set_ylabel(r"$\beta$")
ax[1].set_xlabel("Iteration")

plt.tight_layout()
plt.show()

plot_ld50_hist(ld50_mh, r"MH posterior of $-\alpha/\beta$ given $\beta>0$")

### 2c) ###

with pm.Model() as model:
    alpha = pm.Uniform("alpha", lower=alpha_bounds[0], upper=alpha_bounds[1])
    beta = pm.Uniform("beta", lower=beta_bounds[0], upper=beta_bounds[1])

    p = pm.math.sigmoid(alpha + beta * x)

    pm.Binomial("y_obs", n=n, p=p, observed=y)

    idata = pm.sample(
        draws=3000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=123,
        return_inferencedata=True
    )

print(az.summary(idata, var_names=["alpha", "beta"]))

az.plot_trace(idata, var_names=["alpha", "beta"])
plt.tight_layout()
plt.show()

posterior = az.extract(idata, var_names=["alpha", "beta"]).to_dataframe()
alpha_draws = posterior["alpha"].to_numpy()
beta_draws = posterior["beta"].to_numpy()
draws_pymc = np.column_stack((alpha_draws, beta_draws))

prob_beta_gt_0_pymc, ld50_pymc = summarize_ld50(draws_pymc)

print("P(beta > 0) from PyMC =", prob_beta_gt_0_pymc)
print("LD50 quantiles from PyMC =", np.quantile(ld50_pymc, ld50_quantiles))

plot_ld50_hist(ld50_pymc, r"PyMC posterior of $-\alpha/\beta$ given $\beta>0$")

### 2d) ###

print("\nComparison of Laplace approximation vs full posterior:")
print("P(beta > 0):")
print("  Laplace =", prob_beta_gt_0_laplace)
print("  MH      =", prob_beta_gt_0_mh)
print("  PyMC    =", prob_beta_gt_0_pymc)

print("\nLD50 quantiles:")
print("  Laplace =", np.quantile(ld50_norm, ld50_quantiles))
print("  MH      =", np.quantile(ld50_mh, ld50_quantiles))
print("  PyMC    =", np.quantile(ld50_pymc, ld50_quantiles))
