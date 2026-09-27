"""Shared data for the statistical tests of Appendix E.

Reads ctest_battery.json, ctest_metrics.json and ctest_results.json from the
data folder, ../data beside this script unless CTEST_DATA_DIR says otherwise, and builds what both versions of the tests use: the
difficulty measures of each certified item, the accuracy of each examinee on
each item under full marking, and each examinee's ability.
"""
import json, math, os, statistics as st, collections
import numpy as np
from scipy.optimize import curve_fit

MACHINES = {"Qwen3-14B": ("qwen3-14b-4bit-think", "qwen3-14b"),
            "Qwen3-8B": ("qwen3-8b-4bit-think", "qwen3-8b"),
            "Phi-4-reasoning": ("phi-4-reasoning", "phi-4-reasoning")}

# Reference machines sitting the battery (Appendix D) are not examinees here,
# and gemma-4 was dropped for its timeouts.
EXCLUDED = {"Qwen/Qwen3-14B", "microsoft/Phi-4-reasoning", "gemma-4-26b-a4b-it"}

# Epoch Capability Index of the ten examinees it lists, as read by the ECI
# notebook from Epoch's leaderboard. OLMo-2-7B and Qwen2.5-1.5B are not listed.
ECI = {"gpt-5.6-sol": 161.8, "claude-opus-5-fallback": 162.3,
       "gemini-3.8-flash": 156.5, "gemini-3.6-flash": 154.3,
       "claude-fable-5-1-fallback": 164.5, "claude-sonnet-5-fallback": 156.2,
       "gpt-5-nano": 139.4, "gemini-3.5-flash-lite": 145.1,
       "claude-haiku-4-5-20251001": 142.4, "gemini-3.1-flash-lite": 144.5}

DATA_DIR = os.environ.get("CTEST_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
def data_path(name): return os.path.join(DATA_DIR, name)

B = {b["id"]: b for b in json.load(open(data_path("ctest_battery.json")))["items"]}
MET = json.load(open(data_path("ctest_metrics.json")))["items"]
R = json.load(open(data_path("ctest_results.json")))["rows"]


def is_valid(b):
    return (b.get("valid") is True or
            (b.get("repro", {}).get("seed_cold") or {}).get("verdict") == "reproduced")


def _cert(S):
    P = S.get("programs") or {}
    r = P.get(S.get("p_star")) or {}
    return r.get("cert") or r.get("cert_prev") or {}


def measures(machine):
    """Measures of every item certified under `machine`: Kt, the conditional
    K, log2 tau, the unconditional K, c(x_1:n) and Kc."""
    mid, tag = MACHINES[machine]
    out = {}
    for i, b in B.items():
        if not is_valid(b):
            continue
        S = MET.get(i, {}).get(mid) or {}
        if S.get("band") in (None, "p_lam"):
            continue                      # not certified: the listing is a fallback
        tau = _cert(S).get("tau")
        if S.get("K_cond") is None or not tau:
            continue
        cx = S.get("c_xpref")
        out[i] = dict(Kt=S["K_cond"] + math.log2(tau), Kcond=S["K_cond"],
                      logtau=math.log2(tau), K=b.get(f"K_{tag}"), cx=cx,
                      Kc=(S["K_cond"] + cx) if cx is not None else None)
    return out


def accuracy(machine, items):
    """acc[examinee][item]: mean over attempts of full marking, taking for each
    examinee and item the shortest prefix at or above the machine's L_u."""
    _, tag = MACHINES[machine]
    Lu = {i: B[i].get(f"n_M_{tag}") or 0 for i in items}
    best = {}
    for r in R:
        i, e = r["item"], r["examinee"]
        if i not in Lu or e in EXCLUDED or (r.get("n_M") or 0) < Lu[i]:
            continue
        best[(e, i)] = min(best.get((e, i), 10**9), r["n_M"])
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        k = (r["examinee"], r["item"])
        if k in best and r.get("n_M") == best[k]:
            acc[k[0]][k[1]].append(1.0 if r["n_correct"] / r["k"] >= 0.999 else 0.0)
    # a model whose fallback twin is present is dropped in favour of the twin
    twins = {e[:-len("-fallback")] for e in acc if e.endswith("-fallback")}
    return {e: {i: st.mean(v) for i, v in d.items()}
            for e, d in acc.items() if e not in twins}


def mean_accuracy(acc, items):
    """Accuracy of each item, averaged over the examinees."""
    return {i: st.mean(acc[e][i] for e in acc if i in acc[e]) for i in items}


def _curve(x, th, s):
    return 1.0 / (1.0 + np.exp(np.clip((np.asarray(x) - th) / s, -50, 50)))


def ability(xs, ys):
    """The Kt at which the fitted curve crosses one half, with twice as many
    pseudo-observations at (0, 1) as there are items, as in the paper."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = 2 * len(xs)
    X = np.concatenate([xs, np.zeros(n)]); Y = np.concatenate([ys, np.ones(n)])
    p, _ = curve_fit(_curve, X, Y, p0=[max(np.median(xs), 1.0), 30.0],
                     bounds=([0.1, 1.0], [1e4, 1e4]), maxfev=40000)
    return float(p[0])


def abilities(machine):
    M = measures(machine); items = sorted(M); acc = accuracy(machine, set(items))
    return {e: ability([M[i]["Kt"] for i in items if i in acc[e]],
                       [acc[e][i] for i in items if i in acc[e]]) for e in acc}
