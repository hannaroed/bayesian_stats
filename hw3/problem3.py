import numpy as np
import pandas as pd
from scipy.special import gammaln, psi

# load data
df = pd.read_csv("hw3/data/KidneyCancerClean.csv", skiprows=4)

# Setup:
# Xi = total deaths in 1980-89
# ni = average population in 1980-89
X = (df["dc"] + df["dc.2"]).to_numpy(dtype=float)
n = ((df["pop"] + df["pop.2"]) / 2).to_numpy(dtype=float)

def log_marginal(a, b, X, n):
    if a <= 0 or b <= 0:
        return -np.inf
    N = len(X)
    return np.sum(
        gammaln(X + a)
        + gammaln(n - X + b)
        - gammaln(n + a + b)
    ) + N * (gammaln(a + b) - gammaln(a) - gammaln(b))

def grad_log_marginal(a, b, X, n):
    N = len(X)
    grad_a = np.sum(psi(X + a) - psi(n + a + b)) + N * (psi(a + b) - psi(a))
    grad_b = np.sum(psi(n - X + b) - psi(n + a + b)) + N * (psi(a + b) - psi(b))
    return np.array([grad_a, grad_b])

### 3d) ###
u, v = 0.0, 0.0
max_iter = 10000
tol = 1e-8

for it in range(max_iter):
    a = np.exp(u)
    b = np.exp(v)
    current_val = log_marginal(a, b, X, n)
    g_ab = grad_log_marginal(a, b, X, n)
    g = np.array([a * g_ab[0], b * g_ab[1]])

    grad_norm = np.linalg.norm(g)
    if grad_norm < tol:
        break

    direction = g / grad_norm
    step = 1.0
    improved = False

    while step > 1e-12:
        u_new = u + step * direction[0]
        v_new = v + step * direction[1]
        a_new = np.exp(u_new)
        b_new = np.exp(v_new)
        new_val = log_marginal(a_new, b_new, X, n)

        if np.isfinite(new_val) and new_val > current_val:
            u, v = u_new, v_new
            improved = True
            break

        step *= 0.5

    if not improved:
        break

a = np.exp(u)
b = np.exp(v)

print("Gradient ascent estimate:")
print(f"a_hat = {a:.6f}")
print(f"b_hat = {b:.6f}")
print(f"log marginal likelihood = {log_marginal(a, b, X, n):.6f}")
print(f"iterations = {it+1}")

### 3e) ###
a_grid = np.exp(np.linspace(np.log(1), np.log(100), 400))
b_grid = np.exp(np.linspace(np.log(50000), np.log(300000), 400))

best_val = -np.inf
best_a = None
best_b = None

for a0 in a_grid:
    for b0 in b_grid:
        val = log_marginal(a0, b0, X, n)
        if val > best_val:
            best_val = val
            best_a = a0
            best_b = b0

print("\nGrid search estimate:")
print(f"a_hat_grid = {best_a:.6f}")
print(f"b_hat_grid = {best_b:.6f}")
print(f"log marginal likelihood = {best_val:.6f}")

print("\nComparison:")
print(f"Gradient ascent: a = {a:.6f}, b = {b:.6f}")
print(f"Grid search:      a = {best_a:.6f}, b = {best_b:.6f}")
print(f"abs diff in a = {abs(a - best_a):.6f}")
print(f"abs diff in b = {abs(b - best_b):.6f}")

