from collections import Counter
import re
from scipy.special import gammaln

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())

# read from files
with open("hw3/data/spam.txt", "r", encoding="utf-8") as f:
    spam_text = f.read()

with open("hw3/data/ham.txt", "r", encoding="utf-8") as f:
    ham_text = f.read()

spam_words = tokenize(spam_text)
ham_words = tokenize(ham_text)

spam_counts = Counter(spam_words)
ham_counts = Counter(ham_words)

# combined vocabulary
vocab = sorted(set(spam_counts.keys()) | set(ham_counts.keys()))
K = len(vocab)

n_spam = sum(spam_counts.values())
n_ham = sum(ham_counts.values())

print("K =", K)
print("n_spam =", n_spam)
print("n_ham =", n_ham)


def posterior_means(counts, vocab, alpha):
    n = sum(counts.values())
    denom = n + len(vocab) * alpha
    return {w: (counts.get(w, 0) + alpha) / denom for w in vocab}


def log_posterior_predictive(test_text, train_counts, vocab, alpha):
    test_counts = Counter(tokenize(test_text))
    n = sum(train_counts.values())
    K = len(vocab)
    alpha0 = K * alpha
    m = sum(test_counts.values())

    log_prob = gammaln(n + alpha0) - gammaln(n + alpha0 + m)

    for w, c in test_counts.items():
        Nj = train_counts.get(w, 0)
        log_prob += gammaln(Nj + alpha + c) - gammaln(Nj + alpha)

    return log_prob


def top_k_words(post_mean, k=5):
    return sorted(post_mean.items(), key=lambda x: (-x[1], x[0]))[:k]


alpha = 1.0

### 4a) ###
spam_post_mean = posterior_means(spam_counts, vocab, alpha)

print("\n4a) Top 5 spam words:")
for w, p in top_k_words(spam_post_mean, 5):
    print(f"{w}: {p:.8f}")

### 4b) ###
test1 = "limited time free reward offer claim now"

log_spam_1 = log_posterior_predictive(test1, spam_counts, vocab, alpha)

print("\n4b) log P(test1 | spam) =", log_spam_1)

### 4c) ###
ham_post_mean = posterior_means(ham_counts, vocab, alpha)

print("\n4c) Top 5 ham words:")
for w, p in top_k_words(ham_post_mean, 5):
    print(f"{w}: {p:.8f}")

### 4d) ###
log_ham_1 = log_posterior_predictive(test1, ham_counts, vocab, alpha)
print("\n4d) log P(test1 | ham)  =", log_ham_1)

### 4e) ###
print("\n4e) classification =", "spam" if log_spam_1 > log_ham_1 else "ham")

### 4f) ###
alpha_small = 1e-9
log_spam_1_small = log_posterior_predictive(test1, spam_counts, vocab, alpha_small)
log_ham_1_small = log_posterior_predictive(test1, ham_counts, vocab, alpha_small)

print("\n4f) with alpha = 1e-9")
print("log P(test1 | spam) =", log_spam_1_small)
print("log P(test1 | ham)  =", log_ham_1_small)
print("classification =", "spam" if log_spam_1_small > log_ham_1_small else "ham")

### 4g) ###
test2 = "project update and limited offer"

log_spam_2 = log_posterior_predictive(test2, spam_counts, vocab, alpha)
log_ham_2 = log_posterior_predictive(test2, ham_counts, vocab, alpha)

print("\n4g)")
print("log P(test2 | spam) =", log_spam_2)
print("log P(test2 | ham)  =", log_ham_2)
print("classification =", "spam" if log_spam_2 > log_ham_2 else "ham")
