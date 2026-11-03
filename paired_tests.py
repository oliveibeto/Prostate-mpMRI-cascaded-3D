import numpy as np, pandas as pd
from scipy.stats import norm, binomtest
from sklearn.metrics import roc_auc_score

df = pd.read_csv("stage2_test_predictions.csv")
y = df["label"].to_numpy().astype(int)
p = {"cascade": df["p_cascade"].to_numpy(), "oracle": df["p_oracle"].to_numpy(),
     "baseline": df["p_baseline"].to_numpy()}
thr = {"cascade": 0.517, "oracle": 0.517, "baseline": 0.627}
rng = np.random.RandomState(42)


def delong(y, s1, s2):
    """DeLong test for two correlated AUCs (same cases)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    def comp(s):
        X, Y = s[pos], s[neg]
        psi = (X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])
        return psi.mean(), psi.mean(1), psi.mean(0)
    a1, v10_1, v01_1 = comp(s1)
    a2, v10_2, v01_2 = comp(s2)
    S = np.cov(np.vstack([v10_1, v10_2])) / m + np.cov(np.vstack([v01_1, v01_2])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1 - a2, 2 * norm.sf(abs(z))


def boot_diff(y, s1, s2, B=5000):
    """Paired bootstrap of the AUC difference."""
    d, n = [], len(y)
    for _ in range(B):
        i = rng.randint(0, n, n)
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def mcnemar(y, k1, k2):
    """Exact McNemar test on which cases each model gets right."""
    c1 = (p[k1] >= thr[k1]).astype(int) == y
    c2 = (p[k2] >= thr[k2]).astype(int) == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    pv = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, pv


print("AUC:", {k: round(roc_auc_score(y, v), 3) for k, v in p.items()})
for k in p:
    right = int(((p[k] >= thr[k]).astype(int) == y).sum())
    print(f"{k}: {right}/{len(y)} correct, exact binomial p vs 50% (two-sided) = "
          f"{binomtest(right, len(y), 0.5).pvalue:.3f}")

for a, b in [("cascade", "baseline"), ("cascade", "oracle")]:
    diff, pd_ = delong(y, p[a], p[b])
    lo, hi, pb = boot_diff(y, p[a], p[b])
    n1, n2, pm = mcnemar(y, a, b)
    print(f"\n{a} vs {b}")
    print(f"  AUC difference {diff:+.3f} | DeLong p = {pd_:.3f}")
    print(f"  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] | bootstrap p = {pb:.3f}")
    print(f"  McNemar: {a} right/{b} wrong = {n1}, reverse = {n2}, exact p = {pm:.3f}")