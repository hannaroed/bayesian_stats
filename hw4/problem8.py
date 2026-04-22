import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

df = pd.read_csv("hw4/data/spam7.csv")

df["y"] = (df["yesno"] == "y").astype(int)

df["x1"] = np.log(df["crl.tot"])
df["x2"] = np.log(df["dollar"] + 0.001)
df["x3"] = np.log(df["bang"] + 0.001)
df["x4"] = np.log(df["money"] + 0.001)
df["x5"] = np.log(df["n000"] + 0.001)
df["x6"] = np.log(df["make"] + 0.001)

X = df[["x1", "x2", "x3", "x4", "x5", "x6"]].to_numpy()
y = df["y"].to_numpy()

with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma=100)
    beta = pm.Normal("beta", mu=0, sigma=100, shape=6)

    eta = beta0 + pm.math.dot(X, beta)
    p = pm.math.sigmoid(eta)

    obs = pm.Bernoulli("obs", p=p, observed=y)

    trace = pm.sample(2000, tune=2000, chains=2, target_accept=0.9, random_seed=42)

summary = az.summary(trace, var_names=["beta0", "beta"], round_to=4)
print(summary)

