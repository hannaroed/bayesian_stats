import numpy as np
import pandas as pd

### 1(a) ###

df = pd.read_csv("hw4/data/spam7.csv")

# response: y=1 for spam, 0 otherwise
y = (df["yesno"] == "y").astype(float).to_numpy()

# design matrix with intercept
X = np.column_stack([
    np.ones(len(df)),
    np.log(df["crl.tot"].to_numpy()),
    np.log(df["dollar"].to_numpy() + 0.001),
    np.log(df["bang"].to_numpy() + 0.001),
    np.log(df["money"].to_numpy() + 0.001),
    np.log(df["n000"].to_numpy() + 0.001),
    np.log(df["make"].to_numpy() + 0.001),
])

print("Shape of X:", X.shape)
print("Number of spam emails:", int(y.sum()))
print("Number of non-spam emails:", int((1-y).sum()))
print("Minimum of crl.tot:", df["crl.tot"].min())

def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def loglik(beta, X, y):
    eta = X @ beta
    return np.sum(y * eta - np.logaddexp(0.0, eta))

def newton_logit(X, y, tol=1e-10, max_iter=100):
    beta = np.zeros(X.shape[1])

    for it in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        w = p * (1 - p)

        grad = X.T @ (y - p)
        XtWX = (X.T * w) @ X

        step = np.linalg.solve(XtWX, grad)
        beta_new = beta + step

        diff = np.max(np.abs(beta_new - beta))
        ll = loglik(beta_new, X, y)

        print(f"iter={it+1:2d}, loglik={ll:.12f}, max_change={diff:.12e}")

        beta = beta_new
        if diff < tol:
            break

    return beta

beta_hat = newton_logit(X, y)
print("\nMLE:")
print(beta_hat)

# Built-in library check
import statsmodels.api as sm

fit = sm.Logit(y, X).fit(disp=0)

print("\nLibrary coefficients:")
print(fit.params)

print("\nMaximum absolute difference from Newton solution:")
print(np.max(np.abs(beta_hat - fit.params)))

### 1(b) ###

eta_hat = X @ beta_hat
p_hat = sigmoid(eta_hat)
W_hat = p_hat * (1 - p_hat)

XtWX_hat = (X.T * W_hat) @ X
post_cov = np.linalg.inv(XtWX_hat)
post_se = np.sqrt(np.diag(post_cov))

terms = [
    "Intercept",
    "log(crl.tot)",
    "log(dollar+0.001)",
    "log(bang+0.001)",
    "log(money+0.001)",
    "log(n000+0.001)",
    "log(make+0.001)"
]

ans = pd.DataFrame({
    "term": terms,
    "beta_hat": beta_hat,
    "posterior_sd": post_se
})

print(ans.round(6).to_string(index=False))

# Built-in library check
print("\nLibrary standard errors:")
print(fit.bse)

print("\nMaximum absolute difference from Laplace posterior sds:")
print(np.max(np.abs(post_se - fit.bse)))
