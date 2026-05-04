### 4e) ###

import numpy as np
import matplotlib.pyplot as plt

N = 5000
sigma = 2.0
rng = np.random.default_rng(238)

def exact_laplace_flow(x0, z0, sigma):
    """
    Computes x(sigma) exactly for the Laplace target
    x0 = starting position
    z0 = starting velocity
    sigma = HMC integration time
    """
    x = float(x0)
    z = float(z0)
    r = float(sigma)

    tol = 1e-14

    while r > 1e-12:

        if abs(x) < tol:
            x = 0.0

            if abs(z) < tol:
                return 0.0

            if z > 0:
                a = -1.0
                h = 2.0 * z
            else:
                a = 1.0
                h = -2.0 * z

        elif x > 0:
            a = -1.0
            h = z + np.sqrt(z**2 + 2.0 * x)

        else:
            a = 1.0
            h = -z + np.sqrt(z**2 - 2.0 * x)

        if r <= h:
            return x + r * z + 0.5 * a * r**2

        else:
            x = 0.0
            z = z + a * h
            r = r - h

    return x

def autocorr_fft(x, max_lag=100):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n]
    acov = acov / np.arange(n, 0, -1)
    acf = acov / acov[0]

    return acf[:max_lag + 1]

def ess_initial_positive(x):
    acf = autocorr_fft(x, max_lag=min(len(x) - 1, 1000))
    n = len(x)

    s = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0:
            break
        s += pair_sum

    tau_int = 1 + 2 * s
    return n / tau_int

def plot_acf(x, title, max_lag=100):
    acf = autocorr_fft(x, max_lag=max_lag)

    plt.figure(figsize=(7, 3))
    plt.bar(np.arange(max_lag + 1), acf)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Run exact HMC
samples = np.empty(N)
x = 0.0

for t in range(N):
    z = rng.normal()
    x = exact_laplace_flow(x, z, sigma)
    samples[t] = x

# Summary
print("Sample mean:", np.mean(samples))
print("Sample variance:", np.var(samples, ddof=1))
print("ESS:", ess_initial_positive(samples))

# Histogram with true density
grid = np.linspace(-8, 8, 500)
true_density = 0.5 * np.exp(-np.abs(grid))

plt.figure(figsize=(7, 4))
plt.hist(samples, bins=40, density=True, alpha=0.6, label="HMC samples")
plt.plot(grid, true_density, label="True Laplace density")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Exact HMC samples from Laplace target")
plt.legend()
plt.tight_layout()
plt.show()

plot_acf(samples, "Autocorrelation plot for exact HMC samples")