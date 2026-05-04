### 1b) ###

import numpy as np
import pandas as pd

names = [
    "Clemente", "F Robinson", "F Howard", "Johnstone", "Berry", "Spencer",
    "Kessinger", "L Alvarado", "Santo", "Swoboda", "Unser", "Williams",
    "Scott", "Petrocelli", "E Rodriguez", "Campaneris", "Munson", "Alvis"
]

H = np.array([18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7], dtype=float)

eos = np.array([
    .346, .298, .276, .222, .273, .270, .263, .210, .269,
    .230, .264, .256, .303, .264, .226, .286, .316, .200
])

n = len(H)

sigma = 1.0
mu0 = 0.0
eta0 = 100.0
alpha = 0.01
beta = 0.01

sqrt45 = np.sqrt(45.0)
y = 2 * sqrt45 * np.arcsin(np.sqrt(H / 45.0))

def avg_from_theta(theta):
    return np.sin(theta / (2 * sqrt45)) ** 2

def gibbs_sampler(y, n_iter=60000, seed=238):
    rng = np.random.default_rng(seed)

    n = len(y)
    sig2 = sigma ** 2
    eta02 = eta0 ** 2

    theta = y.copy()
    mu = np.mean(theta)
    tau2 = np.var(theta) + 1.0

    theta_samp = np.empty((n_iter, n))
    mu_samp = np.empty(n_iter)
    tau_samp = np.empty(n_iter)

    for t in range(n_iter):
        v_theta = 1.0 / (1.0 / sig2 + 1.0 / tau2)
        m_theta = v_theta * (y / sig2 + mu / tau2)
        theta = rng.normal(m_theta, np.sqrt(v_theta), size=n)

        v_mu = 1.0 / (n / tau2 + 1.0 / eta02)
        m_mu = v_mu * (np.sum(theta) / tau2 + mu0 / eta02)
        mu = rng.normal(m_mu, np.sqrt(v_mu))

        shape = alpha + n / 2
        scale = beta + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = 1.0 / rng.gamma(shape, scale=1.0 / scale)

        theta_samp[t] = theta
        mu_samp[t] = mu
        tau_samp[t] = np.sqrt(tau2)

    return theta_samp, mu_samp, tau_samp

N = 60000
B = 10000

theta_g, mu_g, tau_g = gibbs_sampler(y, n_iter=N, seed=238)

theta_g_keep = theta_g[B:]
mu_g_keep = mu_g[B:]
tau_g_keep = tau_g[B:]

avg_g = avg_from_theta(theta_g_keep)

ci_g = np.quantile(avg_g, [0.025, 0.975], axis=0).T
cover_g = (eos >= ci_g[:, 0]) & (eos <= ci_g[:, 1])

gibbs_table = pd.DataFrame({
    "Player": names,
    "EoS average": eos,
    "Gibbs 2.5%": ci_g[:, 0],
    "Gibbs 97.5%": ci_g[:, 1],
    "Contains EoS?": cover_g
})

print(gibbs_table.to_string(index=False))
print("Number of Gibbs intervals containing the actual EoS average:", cover_g.sum(), "out of", n)

### 1c) ###

import matplotlib.pyplot as plt

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
    acf = autocorr_fft(x, max_lag=len(x) - 1)
    N = len(x)

    s = 0.0
    for k in range(1, N - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0:
            break
        s += pair_sum

    tau_int = 1 + 2 * s
    return N / tau_int

def plot_trace(x, title):
    plt.figure(figsize=(8, 3))
    plt.plot(x)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

def plot_acf(x, title, max_lag=100):
    acf = autocorr_fft(x, max_lag=max_lag)
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(max_lag + 1), acf)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.show()

plot_trace(mu_g_keep, "Gibbs trace plot: mu")
plot_acf(mu_g_keep, "Gibbs autocorrelation: mu")

plot_trace(np.log(tau_g_keep), "Gibbs trace plot: log(tau)")
plot_acf(np.log(tau_g_keep), "Gibbs autocorrelation: log(tau)")

ess_mu_g = ess_initial_positive(mu_g_keep)
ess_logtau_g = ess_initial_positive(np.log(tau_g_keep))

print("Gibbs ESS for mu:", ess_mu_g)
print("Gibbs ESS for log(tau):", ess_logtau_g)

### 1d) ###

def logpost_grad(z, y):
    theta = z[:n]
    mu = z[n]
    u = z[n + 1]

    s = np.exp(2 * u)
    A = np.sum((theta - mu) ** 2)

    lp = (
        -0.5 * np.sum((y - theta) ** 2) / sigma ** 2
        -0.5 * A / s
        -(n + 2 * alpha) * u
        -beta / s
        -0.5 * (mu - mu0) ** 2 / eta0 ** 2
    )

    grad = np.empty_like(z)

    grad[:n] = (y - theta) / sigma ** 2 - (theta - mu) / s
    grad[n] = np.sum(theta - mu) / s - (mu - mu0) / eta0 ** 2
    grad[n + 1] = (A + 2 * beta) / s - (n + 2 * alpha)

    return lp, grad

def mala_sampler(y, n_iter=60000, eps=0.12, seed=238):
    rng = np.random.default_rng(seed)

    theta0 = y.copy()
    mu_start = np.mean(y)
    u_start = np.log(np.std(y) + 0.5)

    z = np.r_[theta0, mu_start, u_start]
    d = len(z)

    samples = np.empty((n_iter, d))
    accepted = 0

    lp, grad = logpost_grad(z, y)
    eps2 = eps ** 2

    for t in range(n_iter):
        mean_forward = z + 0.5 * eps2 * grad
        z_prop = mean_forward + eps * rng.normal(size=d)

        lp_prop, grad_prop = logpost_grad(z_prop, y)
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

eps = 0.12
z_m, acc_m = mala_sampler(y, n_iter=N, eps=eps, seed=238)

z_m_keep = z_m[B:]

theta_m_keep = z_m_keep[:, :n]
mu_m_keep = z_m_keep[:, n]
tau_m_keep = np.exp(z_m_keep[:, n + 1])

avg_m = avg_from_theta(theta_m_keep)

ci_m = np.quantile(avg_m, [0.025, 0.975], axis=0).T
cover_m = (eos >= ci_m[:, 0]) & (eos <= ci_m[:, 1])

mala_table = pd.DataFrame({
    "Player": names,
    "EoS average": eos,
    "MALA 2.5%": ci_m[:, 0],
    "MALA 97.5%": ci_m[:, 1],
    "Contains EoS?": cover_m
})

print("MALA step size:", eps)
print("MALA acceptance rate:", acc_m)
print(mala_table.to_string(index=False))
print("Number of MALA intervals containing the actual EoS average:", cover_m.sum(), "out of", n)

plot_trace(mu_m_keep, "MALA trace plot: mu")
plot_acf(mu_m_keep, "MALA autocorrelation: mu")

plot_trace(np.log(tau_m_keep), "MALA trace plot: log(tau)")
plot_acf(np.log(tau_m_keep), "MALA autocorrelation: log(tau)")

ess_mu_m = ess_initial_positive(mu_m_keep)
ess_logtau_m = ess_initial_positive(np.log(tau_m_keep))

print("MALA ESS for mu:", ess_mu_m)
print("MALA ESS for log(tau):", ess_logtau_m)