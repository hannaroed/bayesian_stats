import numpy as np
import math
from scipy import stats

theta = 35.5
sigma = 5.5
alpha = 0.05
M = 100000

rng = np.random.default_rng(0)

cover = 0

for _ in range(M):
    s = 0.0
    n = 0
    while True:
        x = rng.normal(theta, sigma)
        s += x
        n += 1
        if x <= 25:
            break

    xbar = s / n

    # Check whether the confidence interval covers theta
    if n == 1:
        cover += 1
    else:
        tcrit = stats.t.ppf(1 - alpha/2, df=n - 1) # two-sided critical value
        half = sigma * tcrit / math.sqrt(n)
        cover += int(xbar - half <= theta <= xbar + half)

p_hat = cover / M
diff = p_hat - (1 - alpha)

print(f"Estimated probability: {p_hat:.5f}")
print(f"Difference from 1-alpha (= {1-alpha:.2f}): {diff:.5f}")
