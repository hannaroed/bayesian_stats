import numpy as np
import matplotlib.pyplot as plt

### 5a) ###
Q = np.array([
    [4.5, -5.0,  2.0],
    [-5.0, 10.0, -4.0],
    [2.0, -4.0,  2.0]
], dtype=float)

mu_true = np.zeros(3)
Sigma_true = np.linalg.inv(Q)

print("True mean vector:")
print(mu_true)
print("\nTrue covariance matrix:")
print(Sigma_true)

### 5b ###
def gibbs_sampler(N=10000, burn=1000, x0=None, seed=238):
    rng = np.random.default_rng(seed)

    if x0 is None:
        x = np.zeros(3, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    draws = np.zeros((N + burn, 3), dtype=float)

    for t in range(N + burn):
        # x1 | x2, x3 ~ N((5*x2 - 2*x3)/4.5, 2/9)
        x1 = rng.normal(loc=(5*x[1] - 2*x[2]) / 4.5,
                        scale=np.sqrt(2/9))

        # x2 | x1, x3 ~ N((5*x1 + 4*x3)/10, 1/10)
        x2 = rng.normal(loc=(5*x1 + 4*x[2]) / 10.0,
                        scale=np.sqrt(1/10))

        # x3 | x1, x2 ~ N(2*x2 - x1, 1/2)
        x3 = rng.normal(loc=2*x2 - x1,
                        scale=np.sqrt(1/2))

        x[:] = (x1, x2, x3)
        draws[t] = x

    return draws[burn:] # keeping N post-burn-in draws

samples = gibbs_sampler(N=10000, burn=1000, x0=[0, 0, 0], seed=238)

### 5b)(i) ###
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(samples[:, 0], lw=0.7)
axes[0].set_title("Trace plot for $x_1$")
axes[0].set_ylabel("$x_1$")

axes[1].plot(samples[:, 1], lw=0.7)
axes[1].set_title("Trace plot for $x_2$")
axes[1].set_ylabel("$x_2$")

axes[2].plot(samples[:, 2], lw=0.7)
axes[2].set_title("Trace plot for $x_3$")
axes[2].set_ylabel("$x_3$")
axes[2].set_xlabel("Iteration")

plt.tight_layout()
plt.show()

### 5b)(ii) ###
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(samples[:, 0], bins=40, density=True)
axes[0].set_title("Histogram of $x_1$")

axes[1].hist(samples[:, 1], bins=40, density=True)
axes[1].set_title("Histogram of $x_2$")

axes[2].hist(samples[:, 2], bins=40, density=True)
axes[2].set_title("Histogram of $x_3$")

plt.tight_layout()
plt.show()

### 5b)(iii) ###
sample_mean = samples.mean(axis=0)
sample_cov = np.cov(samples, rowvar=False)

print("\nEmpirical mean vector:")
print(sample_mean)

print("\nEmpirical covariance matrix:")
print(sample_cov)

### 5b)(iv) ###
print("\nDifference: empirical mean - true mean")
print(sample_mean - mu_true)

print("\nDifference: empirical covariance - true covariance")
print(sample_cov - Sigma_true)

