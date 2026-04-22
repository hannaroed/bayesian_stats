import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

### 2(c) ###

frogs = pd.read_csv("hw4/data/frogs.csv")

if "Unnamed: 0" in frogs.columns:
    frogs = frogs.drop(columns="Unnamed: 0")

y = frogs["pres.abs"].to_numpy().astype(float)

X = pd.DataFrame({
    "altitude": frogs["altitude"].to_numpy(),
    "log_distance": np.log(frogs["distance"].to_numpy()),
    "log_NoOfPools": np.log(frogs["NoOfPools"].to_numpy()),
    "NoOfSites": frogs["NoOfSites"].to_numpy(),
    "avrain": frogs["avrain"].to_numpy(),
    "meanmin": frogs["meanmin"].to_numpy(),
    "meanmax": frogs["meanmax"].to_numpy(),
})

X = sm.add_constant(X)
Xmat = X.to_numpy()
col_names = X.columns.tolist()

probit_model = sm.Probit(y, Xmat)
probit_fit = probit_model.fit(disp=False)
beta_hat = probit_fit.params

eta = Xmat @ beta_hat
Phi = np.clip(norm.cdf(eta), 1e-12, 1 - 1e-12)
phi = norm.pdf(eta)

q = (
    -eta * phi * (y - Phi) / (Phi * (1 - Phi))
    - (phi ** 2) / (Phi * (1 - Phi))
    - (phi ** 2) * (y - Phi) * (1 - 2 * Phi) / (Phi ** 2 * (1 - Phi) ** 2)
)

H = Xmat.T @ (q[:, None] * Xmat)
Sigma_laplace = np.linalg.inv(-H)
laplace_sd = np.sqrt(np.diag(Sigma_laplace))

print("Probit MLE beta_hat")
for name, val in zip(col_names, beta_hat):
    print(f"{name}: {val:.6f}")

print("\nLaplace posterior covariance matrix")
print(Sigma_laplace)

print("\nLaplace posterior standard deviations")
for name, val in zip(col_names, laplace_sd):
    print(f"{name}: {val:.6f}")

### 2(d) ###

library_se = probit_fit.bse

print("Library standard errors from probit fit")
for name, val in zip(col_names, library_se):
    print(f"{name}: {val:.6f}")

print("\nComparison of Laplace posterior SDs and library SEs")
for name, sd1, sd2 in zip(col_names, laplace_sd, library_se):
    print(f"{name}: Laplace SD = {sd1:.6f}, Library SE = {sd2:.6f}, Difference = {abs(sd1 - sd2):.6e}")

print(f"\nMaximum absolute difference = {np.max(np.abs(laplace_sd - library_se)):.6e}")

