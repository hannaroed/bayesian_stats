### 2a) ###

import numpy as np
import matplotlib.pyplot as plt

N = 50000
d = 9
rng = np.random.default_rng(238)

# Direct samples from the model
v_direct = rng.normal(loc=0, scale=3, size=N)
x_direct = rng.normal(
    loc=0,
    scale=np.exp(v_direct[:, None] / 2),
    size=(N, d)
)

print("Direct sampling:")
print("Sample mean of v:", np.mean(v_direct))
print("Sample variance of v:", np.var(v_direct, ddof=1))

plt.figure(figsize=(6, 4))
plt.scatter(v_direct, x_direct[:, 0], s=4, alpha=0.25)
plt.xlabel("v")
plt.ylabel("x1")
plt.title("Direct samples from Neal's funnel: (v, x1)")
plt.tight_layout()
plt.show()

### 2b) ###

def autocorr_fft(x, max_lag=100):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)

    size = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:N]
    acov = acov / np.arange(N, 0, -1)
    acf = acov / acov[0]

    return acf[:max_lag + 1]

def ess_initial_positive(x):
    acf = autocorr_fft(x, max_lag=min(len(x) - 1, 5000))
    N = len(x)

    s = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0:
            break
        s += pair_sum

    return N / (1 + 2 * s)

def plot_acf(x, title, max_lag=100):
    acf = autocorr_fft(x, max_lag=max_lag)

    plt.figure(figsize=(7, 3))
    plt.bar(np.arange(max_lag + 1), acf)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def logpost_grad_original(z):
    v = z[0]
    x = z[1:]

    S = np.sum(x ** 2)

    lp = -v ** 2 / 18 - (d / 2) * v - 0.5 * np.exp(-v) * S

    grad = np.empty_like(z)
    grad[0] = -v / 9 - d / 2 + 0.5 * np.exp(-v) * S
    grad[1:] = -np.exp(-v) * x

    return lp, grad

def mala_sampler(logpost_grad, z0, n_iter=50000, eps=1.0, seed=238):
    rng = np.random.default_rng(seed)

    z = np.array(z0, dtype=float)
    dim = len(z)

    samples = np.empty((n_iter, dim))
    accepted = 0

    lp, grad = logpost_grad(z)
    eps2 = eps ** 2

    for t in range(n_iter):
        mean_forward = z + 0.5 * eps2 * grad
        z_prop = mean_forward + eps * rng.normal(size=dim)

        lp_prop, grad_prop = logpost_grad(z_prop)
        mean_reverse = z_prop + 0.5 * eps2 * grad_prop

        log_q_forward = -0.5 * np.sum((z_prop - mean_forward) ** 2) / eps2
        log_q_reverse = -0.5 * np.sum((z - mean_reverse) ** 2) / eps2

        log_accept = lp_prop - lp + log_q_reverse - log_q_forward

        if np.log(rng.uniform()) < log_accept:
            z = z_prop
            lp = lp_prop
            grad = grad_prop
            accepted += 1

        samples[t] = z

    return samples, accepted / n_iter

# Run MALA on original variables
z0 = np.zeros(d + 1)
samples_original, acc_original = mala_sampler(
    logpost_grad_original,
    z0,
    n_iter=N,
    eps=1.0,
    seed=238
)

v_original = samples_original[:, 0]
x_original = samples_original[:, 1:]

print("MALA on original variables:")
print("Acceptance rate:", acc_original)
print("Sample mean of v:", np.mean(v_original))
print("Sample variance of v:", np.var(v_original, ddof=1))
print("ESS for v:", ess_initial_positive(v_original))

plt.figure(figsize=(6, 4))
plt.scatter(v_original, x_original[:, 0], s=4, alpha=0.25)
plt.xlabel("v")
plt.ylabel("x1")
plt.title("MALA samples on original variables: (v, x1)")
plt.tight_layout()
plt.show()

plot_acf(v_original, "Autocorrelation plot for v: original MALA")

### 2c) ###

def logpost_grad_transformed(z):
    v = z[0]
    xtilde = z[1:]

    lp = -v ** 2 / 18 - 0.5 * np.sum(xtilde ** 2)

    grad = np.empty_like(z)
    grad[0] = -v / 9
    grad[1:] = -xtilde

    return lp, grad

# Run MALA on transformed variables
z0 = np.zeros(d + 1)
samples_transformed, acc_transformed = mala_sampler(
    logpost_grad_transformed,
    z0,
    n_iter=N,
    eps=1.0,
    seed=238
)

v_transformed = samples_transformed[:, 0]
xtilde_samples = samples_transformed[:, 1:]

# Convert back to original x variables
x_from_transformed = np.exp(v_transformed[:, None] / 2) * xtilde_samples

print("MALA on transformed variables:")
print("Acceptance rate:", acc_transformed)
print("Sample mean of v:", np.mean(v_transformed))
print("Sample variance of v:", np.var(v_transformed, ddof=1))
print("ESS for v:", ess_initial_positive(v_transformed))

plt.figure(figsize=(6, 4))
plt.scatter(v_transformed, x_from_transformed[:, 0], s=4, alpha=0.25)
plt.xlabel("v")
plt.ylabel("x1")
plt.title("Transformed MALA converted back to original variables: (v, x1)")
plt.tight_layout()
plt.show()

plot_acf(v_transformed, "Autocorrelation plot for v: transformed MALA")