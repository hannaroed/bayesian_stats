import numpy as np

### 4(b) ###

# given
x = 0
n = 1295
mu = -9.27308950
sigma = np.sqrt(0.09259668)

# grid on logit scale
k = 12 # +/- k sigma
dz = 2e-4 # step size
z = np.arange(mu - k*sigma, mu + k*sigma + dz, dz)
theta = 1 / (1 + np.exp(-z))

# posterior weights
logw = -0.5*((z - mu)/sigma)**2 + x*np.log(theta) + (n-x)*np.log1p(-theta)
logw -= logw.max()
w = np.exp(logw)
w /= w.sum()

# posterior mean, variance, and credible interval
mean_theta = np.sum(theta * w)
var_theta = np.sum(theta**2 * w) - mean_theta**2
cdf = np.cumsum(w)
ci_lo, ci_hi = np.interp([0.025, 0.975], cdf, theta)

print(f"Posterior mean        = {mean_theta:.10e}")
print(f"Posterior variance    = {var_theta:.10e}")
print(f"95% credible interval = [{ci_lo:.10e}, {ci_hi:.10e}]")

import pymc as pm

### 4(c) ###

# given
x = 0
n = 1295
mu = -9.27308950
sigma = np.sqrt(0.09259668)

with pm.Model():
    z = pm.Normal("z", mu=mu, sigma=sigma)
    theta = pm.Deterministic("theta", pm.math.sigmoid(z))
    pm.Binomial("xobs", n=n, p=theta, observed=x)

    idata = pm.sample(
        draws=2000, tune=1000,
        chains=2, target_accept=0.95,
        random_seed=0
    )

draws = idata.posterior["theta"].values.reshape(-1)
mean_theta = draws.mean()
var_theta = draws.var()
ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])

print(f"Posterior mean (PyMC)        = {mean_theta:.10e}")
print(f"Posterior variance (PyMC)    = {var_theta:.10e}")
print(f"95% credible interval (PyMC) = [{ci_lo:.10e}, {ci_hi:.10e}]")
