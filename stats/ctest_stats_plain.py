"""The statistical tests of Appendix E, in plain Python.

This is the version the table in the paper was first computed with. It uses
only the standard library for the tests; the abilities are fitted with
SciPy's curve_fit, through ctest_data.
"""
import math, random, statistics as st
from ctest_data import (MACHINES, ECI, measures, accuracy, mean_accuracy, abilities)

N_BOOT, N_PERM_ITEMS, N_PERM_EXAMINEES = 4000, 10000, 20000


def fmt(p):
    """p to four decimals, or < 0.0001 at the resolution of the test."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"
random.seed(0)


def ranks(v):
    """Ranks with ties given their average rank."""
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for q in range(i, j + 1):
            r[order[q]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def bootstrap_ci(x, y, f=spearman):
    """Percentile interval over items resampled with replacement."""
    idx = range(len(x)); vals = []
    for _ in range(N_BOOT):
        s = [random.choice(idx) for _ in idx]
        vals.append(f([x[i] for i in s], [y[i] for i in s]))
    vals.sort()
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def permutation_p(x, y, f=spearman, n=N_PERM_ITEMS, sign=-1):
    """One-sided p in the predicted direction: sign=-1 for a negative
    correlation, +1 for a positive one. Counted as (b + 1) / (n + 1)."""
    r0 = f(x, y); b = 0
    for _ in range(n):
        r = f(x, random.sample(y, len(y)))
        b += (r <= r0) if sign < 0 else (r >= r0)
    return (b + 1) / (n + 1)


def fisher_ci(r, n):
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval, by bisection on the binomial tail."""
    def cdf(x, p):
        return sum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(x + 1))
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) < alpha / 2)
    high = 1.0 if k == n else solve(lambda p: cdf(k, p) > alpha / 2)
    return low, high


print("Spearman correlation of Kt with accuracy, over items")
for m in MACHINES:
    M = measures(m); items = sorted(M); acc = accuracy(m, set(items)); y = mean_accuracy(acc, items)
    X = [M[i]["Kt"] for i in items]; Y = [y[i] for i in items]
    lo, hi = bootstrap_ci(X, Y)
    print(f"  {m:16s} {len(items)} items  {spearman(X, Y):+.2f} [{lo:+.2f}, {hi:+.2f}]  p {fmt(permutation_p(X, Y))}")

print("\nQwen3-14B: Kt against the other measures, paired bootstrap of the difference")
M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items)); y = mean_accuracy(acc, items)
for k in ("K", "cx", "logtau", "Kcond", "Kc"):
    it = [i for i in items if M[i][k] is not None]
    d = []
    for _ in range(N_BOOT):
        s = [random.choice(it) for _ in it]
        d.append(spearman([M[i]["Kt"] for i in s], [y[i] for i in s]) -
                 spearman([M[i][k] for i in s], [y[i] for i in s]))
    d.sort()
    diff = spearman([M[i]["Kt"] for i in it], [y[i] for i in it]) - spearman([M[i][k] for i in it], [y[i] for i in it])
    print(f"  Kt minus {k:7s} {diff:+.2f} [{d[int(.025*N_BOOT)]:+.2f}, {d[int(.975*N_BOOT)-1]:+.2f}]")

print("\nQwen3-14B: each examinee separately")
for e in sorted(acc):
    it = [i for i in items if i in acc[e]]
    X = [M[i]["Kt"] for i in it]; Y = [acc[e][i] for i in it]
    p = permutation_p(X, Y)
    print(f"  {e.split('/')[-1][:26]:26s} {spearman(X, Y):+.2f}  p {fmt(p)}  {'below' if p < 0.05/12 else 'NOT below'} 0.05/12")

TH = {m: abilities(m) for m in MACHINES}
print("\nAbility against ECI, Qwen3-14B")
ks = [e for e in TH["Qwen3-14B"] if e in ECI]
x = [TH["Qwen3-14B"][e] for e in ks]; y = [ECI[e] for e in ks]
r = pearson(x, y); lo, hi = fisher_ci(r, len(ks))
print(f"  Pearson  {r:+.2f} [{lo:+.2f}, {hi:+.2f}]  p {fmt(permutation_p(x, y, pearson, N_PERM_EXAMINEES, +1))}  over {len(ks)}")
print(f"  Spearman {spearman(x, y):+.2f}  p {fmt(permutation_p(x, y, spearman, N_PERM_EXAMINEES, +1))}")

print("\nAgreement of the ability rankings, 12 examinees")
for a, b in (("Qwen3-14B", "Qwen3-8B"), ("Qwen3-14B", "Phi-4-reasoning"), ("Qwen3-8B", "Phi-4-reasoning")):
    ks = sorted(set(TH[a]) & set(TH[b]))
    x = [TH[a][e] for e in ks]; y = [TH[b][e] for e in ks]
    print(f"  {a} and {b}: {spearman(x, y):+.2f}  p {fmt(permutation_p(x, y, spearman, N_PERM_EXAMINEES, +1))}")

print("\ngpt5sol on the items above 190 bits, Qwen3-14B")
hi_items = [i for i in items if M[i]["Kt"] > 190]
solved = sum(1 for i in hi_items if acc["gpt-5.6-sol"].get(i, 0) > 0.5)
lo, hi = clopper_pearson(solved, len(hi_items))
print(f"  solved on a majority of attempts: {solved} of {len(hi_items)} [{lo:.0%}, {hi:.0%}]")
