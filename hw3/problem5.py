import re
import numpy as np
from collections import Counter
from scipy.special import gammaln, psi
from scipy.optimize import minimize

# load + tokenize
with open("hw3/data/alice.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# treat period as a separate token
# keep alphabetic words and periods only
tokens = re.findall(r"[a-z]+|\.", text)

n = len(tokens)
print("Number of tokens n =", n)

# vocabulary
vocab = sorted(set(tokens))
K = len(vocab)
print("Vocabulary size K =", K)

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

### 5a) ###

print("First 50 tokens:")
print(tokens[:50])

#### 5b) ###

print("K =", K)
print("First 30 vocabulary items:")
print(vocab[:30])

### 5c) i) ###
alpha = 1e-3

token_counts = Counter(tokens)

# posterior: Dirichlet(N_1 + alpha, ..., N_K + alpha)
post_mean_iid = {
    w: (token_counts[w] + alpha) / (n + K * alpha)
    for w in vocab
}

top5_iid = sorted(post_mean_iid.items(), key=lambda x: (-x[1], x[0]))[:5]

print("Top 5 posterior mean tokens under i.i.d. model:")
for w, p in top5_iid:
    print(f"{w}: {p:.8f}")

### 5c) ii) ###
rng = np.random.default_rng(238)

# sample one p vector from posterior
alpha_post = np.array([token_counts[w] + alpha for w in vocab], dtype=float)
p_draw = rng.dirichlet(alpha_post)

def generate_iid_sentence(vocab, p_draw, rng, max_len=100):
    sent = []
    while True:
        w = rng.choice(vocab, p=p_draw)
        sent.append(w)
        if w == "." or len(sent) >= max_len:
            break
    return " ".join(sent)

print("20 generated sentences from i.i.d. model:\n")
for s in range(20):
    print(f"{s+1:2d}. {generate_iid_sentence(vocab, p_draw, rng)}")

### 5d) ###

# build bigram counts x_{j|i}
X = np.zeros((K, K), dtype=int)

for t in range(n - 1):
    i = word_to_idx[tokens[t]]
    j = word_to_idx[tokens[t + 1]]
    X[i, j] += 1

x_row = X.sum(axis=1) # x_i
print("Bigram count matrix shape:", X.shape)


def log_evidence(a, X):
    # a is length-K, all positive
    if np.any(a <= 0):
        return -np.inf

    A = np.sum(a)
    x_row = X.sum(axis=1)

    val = 0.0
    for i in range(X.shape[0]):
        val += gammaln(A) - gammaln(x_row[i] + A)
        val += np.sum(gammaln(X[i, :] + a) - gammaln(a))
    return val


def grad_log_evidence(a, X):
    A = np.sum(a)
    x_row = X.sum(axis=1)

    common = np.sum(psi(A) - psi(x_row + A)) # scalar summed over i
    grad = np.empty_like(a)

    for j in range(len(a)):
        grad[j] = common + np.sum(psi(X[:, j] + a[j]) - psi(a[j]))

    return grad


# optimize over eta = log(a), so a = exp(eta) stays positive
def objective_eta(eta, X):
    a = np.exp(eta)
    return -log_evidence(a, X)

def grad_objective_eta(eta, X):
    a = np.exp(eta)
    grad_a = grad_log_evidence(a, X)
    return -(grad_a * a) # chain rule


eta0 = np.full(K, np.log(1e-3))

res = minimize(
    objective_eta,
    eta0,
    args=(X,),
    jac=grad_objective_eta,
    method="L-BFGS-B",
    options={"maxiter": 500}
)

a_hat = np.exp(res.x)

print("Optimization success:", res.success)
print("Message:", res.message)
print("First 20 estimated a_j values:")
print(a_hat[:20])

### 5e) ###

A_hat = np.sum(a_hat)
row_totals = X.sum(axis=1)

# posterior mean transition matrix
P_hat = np.zeros((K, K), dtype=float)
for i in range(K):
    P_hat[i, :] = (X[i, :] + a_hat) / (row_totals[i] + A_hat)

targets = ["alice", "she", "the"]

for token in targets:
    if token not in word_to_idx:
        print(f"Token '{token}' not in vocabulary.")
        continue

    i = word_to_idx[token]
    top4_idx = np.argsort(P_hat[i, :])[::-1][:4]

    print(f"\nTop 4 next tokens after '{token}':")
    for j in top4_idx:
        print(f"{idx_to_word[j]}: {P_hat[i, j]:.8f}")

### 5f) ###

rng = np.random.default_rng(238)

period_idx = word_to_idx["."]

def generate_ar1_sentence(P_hat, idx_to_word, start_idx, rng, max_len=100):
    current = start_idx
    sent = []

    while True:
        next_idx = rng.choice(np.arange(P_hat.shape[1]), p=P_hat[current, :])
        next_word = idx_to_word[next_idx]
        sent.append(next_word)

        if next_word == "." or len(sent) >= max_len:
            break

        current = next_idx

    return " ".join(sent)

print("20 generated sentences from AR(1) model:\n")
for s in range(20):
    print(f"{s+1:2d}. {generate_ar1_sentence(P_hat, idx_to_word, period_idx, rng)}")
