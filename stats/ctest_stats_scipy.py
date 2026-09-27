"""The statistical tests of Appendix E, with SciPy.

The same tests as ctest_stats_plain.py, using scipy.stats: spearmanr and
pearsonr for the correlations, bootstrap with the percentile method for the
intervals over items, permutation_test for the one-sided p-values, and
binomtest with the exact (Clopper-Pearson) interval. The Fisher interval for a
Pearson correlation is the arctanh transformation.
"""
import numpy as np
from scipy import stats
from ctest_data import (MACHINES, ECI, measures, accuracy, mean_accuracy, abilities)

N_BOOT, N_PERM_ITEMS, N_PERM_EXAMINEES = 4000, 10000, 20000


def fmt(p):
    """p to four decimals, or < 0.0001 at the resolution of the test."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"
rng = np.random.default_rng(0)


def rho(a, b):
    return stats.spearmanr(a, b).statistic


def bootstrap_ci(x, y, f=rho):
    res = stats.bootstrap((x, y), f, paired=True, vectorized=False, n_resamples=N_BOOT,
                          method="percentile", random_state=rng)
    return res.confidence_interval.low, res.confidence_interval.high


def permutation_p(x, y, f=rho, n=N_PERM_ITEMS, alternative="less"):
    y = np.asarray(y)
    res = stats.permutation_test((x,), lambda a: f(a, y), permutation_type="pairings",
                                 n_resamples=n, alternative=alternative, random_state=rng)
    return res.pvalue


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


print("Spearman correlation of Kt with accuracy, over items")
for m in MACHINES:
    M = measures(m); items = sorted(M); acc = accuracy(m, set(items)); y = mean_accuracy(acc, items)
    X = np.array([M[i]["Kt"] for i in items]); Y = np.array([y[i] for i in items])
    lo, hi = bootstrap_ci(X, Y)
    print(f"  {m:16s} {len(items)} items  {rho(X, Y):+.2f} [{lo:+.2f}, {hi:+.2f}]  p {fmt(permutation_p(X, Y))}")

print("\nQwen3-14B: Kt against the other measures, paired bootstrap of the difference")
M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items)); y = mean_accuracy(acc, items)
for k in ("K", "cx", "logtau", "Kcond", "Kc"):
    it = [i for i in items if M[i][k] is not None]
    kt = np.array([M[i]["Kt"] for i in it]); ot = np.array([M[i][k] for i in it]); yy = np.array([y[i] for i in it])
    res = stats.bootstrap((kt, ot, yy), lambda a, b, c: rho(a, c) - rho(b, c), paired=True,
                          vectorized=False, n_resamples=N_BOOT, method="percentile", random_state=rng)
    print(f"  Kt minus {k:7s} {rho(kt, yy) - rho(ot, yy):+.2f} "
          f"[{res.confidence_interval.low:+.2f}, {res.confidence_interval.high:+.2f}]")

print("\nQwen3-14B: each examinee separately")
for e in sorted(acc):
    it = [i for i in items if i in acc[e]]
    X = np.array([M[i]["Kt"] for i in it]); Y = np.array([acc[e][i] for i in it])
    p = permutation_p(X, Y)
    print(f"  {e.split('/')[-1][:26]:26s} {rho(X, Y):+.2f}  p {fmt(p)}  {'below' if p < 0.05/12 else 'NOT below'} 0.05/12")

TH = {m: abilities(m) for m in MACHINES}
print("\nAbility against ECI, Qwen3-14B")
ks = [e for e in TH["Qwen3-14B"] if e in ECI]
x = np.array([TH["Qwen3-14B"][e] for e in ks]); y = np.array([ECI[e] for e in ks])
r = stats.pearsonr(x, y).statistic; lo, hi = fisher_ci(r, len(ks))
pear = lambda a, b: stats.pearsonr(a, b).statistic
print(f"  Pearson  {r:+.2f} [{lo:+.2f}, {hi:+.2f}]  p {fmt(permutation_p(x, y, pear, N_PERM_EXAMINEES, 'greater'))}  over {len(ks)}")
print(f"  Spearman {rho(x, y):+.2f}  p {fmt(permutation_p(x, y, rho, N_PERM_EXAMINEES, 'greater'))}")

print("\nAgreement of the ability rankings, 12 examinees")
for a, b in (("Qwen3-14B", "Qwen3-8B"), ("Qwen3-14B", "Phi-4-reasoning"), ("Qwen3-8B", "Phi-4-reasoning")):
    ks = sorted(set(TH[a]) & set(TH[b]))
    x = np.array([TH[a][e] for e in ks]); y = np.array([TH[b][e] for e in ks])
    print(f"  {a} and {b}: {rho(x, y):+.2f}  p {fmt(permutation_p(x, y, rho, N_PERM_EXAMINEES, 'greater'))}")

print("\ngpt5sol on the items above 190 bits, Qwen3-14B")
hi_items = [i for i in items if M[i]["Kt"] > 190]
solved = sum(1 for i in hi_items if acc["gpt-5.6-sol"].get(i, 0) > 0.5)
ci = stats.binomtest(solved, len(hi_items)).proportion_ci(method="exact")
print(f"  solved on a majority of attempts: {solved} of {len(hi_items)} [{ci.low:.0%}, {ci.high:.0%}]")
