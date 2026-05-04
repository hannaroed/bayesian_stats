### 3c) ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 10000
sigma = 1.0

def logpi(x):
    return -0.5 * x**2

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

    tau_int = 1 + 2 * s
    return N / tau_int

def lag1_autocorr(x):
    return autocorr_fft(x, max_lag=1)[1]

def simulate_chain(N=10000, sigma=1.0, method="alpha", seed=238):
    rng = np.random.default_rng(seed)

    x = 0.0
    samples = np.empty(N)
    accepted = 0

    for t in range(N):
        y = x + sigma * rng.normal()

        log_r = logpi(y) - logpi(x)

        if method == "alpha":
            accept_prob = min(1.0, np.exp(log_r))

        elif method == "beta":
            # beta = r / (1 + r), computed stably
            if log_r >= 0:
                accept_prob = 1.0 / (1.0 + np.exp(-log_r))
            else:
                r = np.exp(log_r)
                accept_prob = r / (1.0 + r)

        else:
            raise ValueError("method must be 'alpha' or 'beta'")

        if rng.uniform() < accept_prob:
            x = y
            accepted += 1

        samples[t] = x

    acceptance_rate = accepted / N

    return samples, acceptance_rate

# Run chains
samples_alpha, acc_alpha = simulate_chain(
    N=N,
    sigma=sigma,
    method="alpha",
    seed=238
)

samples_beta, acc_beta = simulate_chain(
    N=N,
    sigma=sigma,
    method="beta",
    seed=239
)

# Summary table
results = pd.DataFrame({
    "Chain": ["Usual MH alpha", "Alternative beta"],
    "Acceptance rate": [acc_alpha, acc_beta],
    "Sample mean": [np.mean(samples_alpha), np.mean(samples_beta)],
    "Sample variance": [
        np.var(samples_alpha, ddof=1),
        np.var(samples_beta, ddof=1)
    ],
    "Lag-1 autocorrelation": [
        lag1_autocorr(samples_alpha),
        lag1_autocorr(samples_beta)
    ],
    "ESS": [
        ess_initial_positive(samples_alpha),
        ess_initial_positive(samples_beta)
    ]
})

print(results.to_string(index=False))

# Trace plots
plt.figure(figsize=(8, 3))
plt.plot(samples_alpha)
plt.xlabel("Iteration")
plt.ylabel("X")
plt.title("Trace plot: usual MH alpha")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(samples_beta)
plt.xlabel("Iteration")
plt.ylabel("X")
plt.title("Trace plot: alternative beta")
plt.tight_layout()
plt.show()