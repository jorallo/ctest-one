"""Further analyses reported in the appendices of the paper, with SciPy.

Run all of them, or name the ones you want:

    python ctest_stats_more.py                  # every section
    python ctest_stats_more.py ratio kt_min     # only these

Sections:
    prior        sensitivity of abilities and windows to the prior of the fit
    ratio        abilities under the three machines: correlations and proportionality
    kt_min       the smallest Kt over the machines that certify an item
    tau          what the time term log2(tau) contributes, by band of Kt
    upper        the items above 190 bits, and accuracy by band of Kt
    seed         seed length against accuracy, and against Kt
    budget       abilities with cut replies left out, and where the cuts fall
    refusals     items on which the plain Anthropic models refused
    judge        the judge against a mechanical check, and residuals against ECI
    rivals       how close six composed rivals are to saturation
    supra        each examinee in its own supra window, with exact intervals
    misspecified the examinees' shared wrong answers, and results without the
                 items that have a missed rival

Reads the data folder through ctest_data.py (../data unless CTEST_DATA_DIR says
otherwise), including the three ctest_pricing_<machine>.json files. The random seeds are fixed, so each run gives the same result.
"""
import sys, re, json, math, collections, statistics as st
from math import comb
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from ctest_data import MACHINES, ECI, EXCLUDED, B, MET, R, is_valid, measures, accuracy, mean_accuracy, abilities, data_path

RNG = np.random.default_rng(0)
STRONG = ['gpt-5.6-sol', 'claude-opus-5-fallback', 'gemini-3.8-flash',
          'claude-fable-5-1-fallback', 'gemini-3.6-flash', 'claude-sonnet-5-fallback']
FLAGGED = {"Qwen3-14B": {'v17-0133', 'v17-0110', 'v17-0127'},
           "Qwen3-8B": {'v17-0133', 'v17-0002', 'v17-0130'},
           "Phi-4-reasoning": {'v17-0133', 'v17-0045', 'v17-0002', 'v17-0130'}}
PLAIN_ANTHROPIC = ('claude-opus-5', 'claude-sonnet-5', 'claude-fable-5-1')


def head(t): print(f"\n=== {t}")
def rho(a, b): return stats.spearmanr(a, b).statistic
def short(e): return e.split('/')[-1][:26]


def curve(x, th, s): return 1.0 / (1.0 + np.exp(np.clip((np.asarray(x) - th) / s, -50, 50)))


def fit(xs, ys, mult=2.0):
    """theta and s of the logistic with mult*N pseudo-observations at (0, 1)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float); n = int(round(mult * len(xs)))
    p, _ = curve_fit(curve, np.r_[xs, np.zeros(n)], np.r_[ys, np.ones(n)],
                     p0=[max(np.median(xs), 1.0), 30.0], bounds=([0.1, 1.0], [1e4, 1e4]), maxfev=40000)
    return p


def rows_at(machine, items):
    """Result rows at the shortest prefix at or above the machine's L_u, twins removed."""
    _, tag = MACHINES[machine]; Lu = {i: B[i].get(f'n_M_{tag}') or 0 for i in items}; best = {}
    for r in R:
        i, e = r['item'], r['examinee']
        if i in Lu and e not in EXCLUDED and (r.get('n_M') or 0) >= Lu[i]:
            best[(e, i)] = min(best.get((e, i), 10**9), r['n_M'])
    twins = {e[:-9] for (e, _) in best if e.endswith('-fallback')}
    return [r for r in R if (r['examinee'], r['item']) in best
            and r.get('n_M') == best[(r['examinee'], r['item'])] and r['examinee'] not in twins]


def full(r): return 1.0 if r['n_correct'] == r['k'] else 0.0


# ------------------------------------------------------------------------ sections
def s_prior():
    head("prior: abilities and windows by number of prior points (Qwen3-14B)")
    M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items))
    res = {}
    for m in (0, 0.5, 1, 2, 4):
        th, hi = {}, {}
        for e in acc:
            it = [i for i in items if i in acc[e]]
            t, s = fit([M[i]['Kt'] for i in it], [acc[e][i] for i in it], m); th[e] = t; hi[e] = t + s * math.log(2)
        res[m] = (th, hi)
    ref = res[2][0]
    for m in res:
        th = res[m][0]; ks = list(acc); ke = [e for e in ks if e in ECI]
        print(f"  {m:>3}N: largest change {max(abs(th[e]-ref[e]) for e in ks):4.1f} bits, rank vs 2N {rho([th[e] for e in ks],[ref[e] for e in ks]):+.2f}, "
              f"ECI Pearson {stats.pearsonr([th[e] for e in ke],[ECI[e] for e in ke]).statistic:+.2f}, gpt5sol supra start {res[m][1]['gpt-5.6-sol']:.0f}")


def s_ratio():
    head("ratio: abilities under the three machines")
    TH = {m: abilities(m) for m in MACHINES}; names = list(MACHINES)
    for label, drop_small in (("12 examinees", False), ("without the two small models", True)):
        ks = sorted(set.intersection(*[set(v) for v in TH.values()]))
        if drop_small: ks = [e for e in ks if not e.startswith(('allenai/', 'Qwen/Qwen2.5'))]
        X = {m: np.array([TH[m][e] for e in ks]) for m in names}
        print(f"  {label}: Pearson matrix")
        for a in names: print("     " + "  ".join(f"{stats.pearsonr(X[a], X[b]).statistic:.3f}" for b in names) + f"   {a}")
        for A, Bm in (("Qwen3-14B", "Qwen3-8B"), ("Qwen3-14B", "Phi-4-reasoning"), ("Qwen3-8B", "Phi-4-reasoning")):
            x, y = X[A], X[Bm]; b, a = np.polyfit(x, y, 1)
            bs = np.array([np.polyfit(x[i], y[i], 1) for i in (RNG.integers(0, len(x), len(x)) for _ in range(4000)) if np.ptp(x[i]) > 0])
            ic = np.percentile(bs[:, 1], [2.5, 97.5]); b0 = (x @ y) / (x @ x)
            r2a = 1 - np.sum((y - (a + b * x))**2) / np.sum((y - y.mean())**2); r2p = 1 - np.sum((y - b0 * x)**2) / np.sum((y - y.mean())**2)
            print(f"     {Bm} on {A}: intercept {a:+.1f} [{ic[0]:+.1f}, {ic[1]:+.1f}], slope {b:.2f}; through the origin {b0:.2f}, R2 {r2p:.3f} against {r2a:.3f}")


def s_kt_min():
    head("kt_min: the smallest Kt over the machines that certify an item")
    MS = {m: measures(m) for m in MACHINES}; union = sorted(set().union(*[set(v) for v in MS.values()]))
    rows = collections.defaultdict(list)
    for r in R:
        if r['item'] in union and r['examinee'] not in EXCLUDED: rows[r['item']].append(r)
    xs, ys, who, nmach = [], [], collections.Counter(), collections.Counter(); per = collections.defaultdict(dict)
    for i in union:
        kt, m = min((MS[m][i]['Kt'], m) for m in MACHINES if i in MS[m]); who[m] += 1
        nmach[sum(1 for mm in MACHINES if i in MS[mm])] += 1
        L = B[i][f"n_M_{MACHINES[m][1]}"]; best = {}
        for r in rows[i]:
            if (r.get('n_M') or 0) >= L: best[r['examinee']] = min(best.get(r['examinee'], 10**9), r['n_M'])
        twins = {e[:-9] for e in best if e.endswith('-fallback')}; d = collections.defaultdict(list)
        for r in rows[i]:
            if r['examinee'] in best and r['examinee'] not in twins and r.get('n_M') == best[r['examinee']]: d[r['examinee']].append(full(r))
        for e, v in d.items(): per[e][i] = (kt, st.mean(v))
        xs.append(kt); ys.append(st.mean(st.mean(v) for v in d.values()))
    print(f"  {len(union)} items; minimum from {dict(who)}; certified by 1/2/3 machines {dict(sorted(nmach.items()))}")
    print(f"  Spearman(min Kt, accuracy) {rho(xs, ys):+.2f}")
    th = {e: fit([k for k, _ in d.values()], [a for _, a in d.values()])[0] for e, d in per.items()}
    ke = [e for e in th if e in ECI]
    print(f"  abilities vs ECI: Pearson {stats.pearsonr([th[e] for e in ke],[ECI[e] for e in ke]).statistic:+.2f}, Spearman {rho([th[e] for e in ke],[ECI[e] for e in ke]):+.2f}")


def s_tau():
    head("tau: log2(tau) by band of Kt, and what it adds (Qwen3-14B)")
    M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items)); y = mean_accuracy(acc, items)
    for a, b in ((0, 50), (50, 100), (100, 150), (150, 190), (190, 400)):
        it = [i for i in items if a <= M[i]['Kt'] < b]
        print(f"  {a:3d}-{b:<3d}: {len(it):2d} items, log2 tau median {st.median(M[i]['logtau'] for i in it):5.1f}, share of Kt median {st.median(M[i]['logtau']/M[i]['Kt'] for i in it):4.0%}")
    med = st.median(M[i]['Kt'] for i in items)
    for name, it in (("lower half", [i for i in items if M[i]['Kt'] <= med]), ("upper half", [i for i in items if M[i]['Kt'] > med])):
        f = lambda k: rho([M[i][k] for i in it], [y[i] for i in it])
        print(f"  {name}: K| {f('Kcond'):+.2f}, Kt {f('Kt'):+.2f}")


def s_upper():
    head("upper: items above 190 bits, and accuracy by band (Qwen3-14B)")
    M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items)); y = mean_accuracy(acc, items)
    ys = {i: st.mean(acc[e][i] for e in STRONG if i in acc[e]) for i in items}
    for i in sorted([i for i in items if M[i]['Kt'] > 190], key=lambda i: -M[i]['Kt']):
        S = MET[i]['qwen3-14b-4bit-think']; tag = (S['programs'][S['p_star']] or {}).get('tag')
        print(f"  {i}  Kt {M[i]['Kt']:5.0f}  r^ {B[i].get('r_hat_qwen3-14b')}  t^ {B[i].get('t_hat_qwen3-14b')}  {tag:10s}  all {y[i]:4.0%}  strong six {ys[i]:4.0%}")
    for a, b in ((0, 50), (50, 100), (100, 150), (150, 190), (190, 400)):
        it = [i for i in items if a <= M[i]['Kt'] < b]
        print(f"  band {a}-{b}: {len(it)} items, all twelve {st.mean(y[i] for i in it):.0%}, strong six {st.mean(ys[i] for i in it):.0%}")


def s_seed():
    head("seed: seed length against accuracy (Qwen3-14B)")
    tag = 'qwen3-14b'; U = [i for i in B if is_valid(B[i]) and B[i].get(f'n_M_{tag}') is not None]
    C = set(measures("Qwen3-14B")); rr = rows_at("Qwen3-14B", set(U)); d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rr: d[r['item']][r['examinee']].append(full(r))
    ya = {i: st.mean(st.mean(v) for v in d[i].values()) for i in U if d[i]}
    for name, it in (("unquestionable", U), ("certified", [i for i in U if i in C]), ("not certified", [i for i in U if i not in C])):
        it = [i for i in it if i in ya]
        print(f"  {name:15s} n={len(it):2d}: seed {rho([len(B[i]['seed']) for i in it],[ya[i] for i in it]):+.2f}, K {rho([B[i].get(f'K_{tag}') for i in it],[ya[i] for i in it]):+.2f}")
    M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items)); y = mean_accuracy(acc, items)
    kt = np.array([M[i]['Kt'] for i in items]); sd = np.array([len(B[i]['seed']) for i in items], float); yy = np.array([y[i] for i in items])
    res = stats.bootstrap((kt, sd, yy), lambda a, b, c: rho(a, c) - rho(b, c), paired=True, vectorized=False, n_resamples=4000, method='percentile', random_state=RNG)
    print(f"  Kt minus seed: {rho(kt,yy)-rho(sd,yy):+.2f} [{res.confidence_interval.low:+.2f}, {res.confidence_interval.high:+.2f}]")


def s_budget():
    head("budget: abilities with cut replies left out (Qwen3-14B)")
    M = measures("Qwen3-14B"); items = set(M); per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows_at("Qwen3-14B", items): per[r['examinee']][r['item']].append((full(r), bool(r.get('was_cut'))))
    for e in sorted(per, key=short):
        ths = []
        for drop in (False, True):
            xs, ys = [], []
            for i, v in per[e].items():
                vals = [s for s, c in v if not (drop and c)]
                if vals: xs.append(M[i]['Kt']); ys.append(st.mean(vals))
            ths.append(fit(xs, ys)[0])
        cut = [M[i]['Kt'] for i, v in per[e].items() for s, c in v if c]
        print(f"  {short(e):26s} cut {len(cut):3d}  theta {ths[0]:4.0f} -> {ths[1]:4.0f}" + (f"   median Kt of cut replies {st.median(cut):.0f}" if cut else ""))


def s_refusals():
    head("refusals: items on which the plain Anthropic models refused (Qwen3-14B)")
    M = measures("Qwen3-14B")
    ref = {r['item'] for r in R if r['examinee'] in PLAIN_ANTHROPIC and str(r.get('finish')).upper() == 'REFUSAL' and r['item'] in M}
    oth = [i for i in M if i not in ref]
    print(f"  {len(ref)} of {len(M)} items; median Kt {st.median(M[i]['Kt'] for i in ref):.0f} against {st.median(M[i]['Kt'] for i in oth):.0f}; above 190 bits {sum(1 for i in ref if M[i]['Kt']>190)}")
    acc = accuracy("Qwen3-14B", set(oth)); y = mean_accuracy(acc, oth)
    print(f"  Spearman(Kt, accuracy) without them: {rho([M[i]['Kt'] for i in oth],[y[i] for i in oth]):+.2f}")


def s_judge():
    head("judge: judge against a mechanical check, and residuals against ECI (Qwen3-14B)")
    M = measures("Qwen3-14B"); norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower().rstrip('.'))
    tab = collections.defaultdict(collections.Counter)
    for r in rows_at("Qwen3-14B", set(M)):
        truth = [norm(t) for t in B[r['item']]['terms'][r['n_M']:r['n_M'] + r['k']]]
        ans = [norm(t) for t in str(r.get('answer') or '').split(',')][:r['k']]
        tab[r['examinee']][(full(r) == 1.0, ans == truth)] += 1
    for e in sorted(tab, key=short):
        c = tab[e]; print(f"  {short(e):26s} judge yes, mechanical no {c[(True,False)]}; judge no, mechanical yes {c[(False,True)]}")
    TH = abilities("Qwen3-14B"); ks = [e for e in TH if e in ECI]
    x = np.array([ECI[e] for e in ks]); y = np.array([TH[e] for e in ks]); b, a = np.polyfit(x, y, 1)
    for e, rr in sorted(zip(ks, y - (a + b * x)), key=lambda z: -z[1]): print(f"  residual {rr:+6.1f}  {e}")


def s_rivals():
    head("rivals: saturation of the six composed rivals")
    for m, (_, tag) in MACHINES.items():
        PR = json.load(open(data_path(f'ctest_pricing_{tag}.json')))['items']; n_items = 0; nr = 0; nb = 0; only = 0; shares = []
        exp = {k: 0.0 for k in (1, 3, 4, 5)}; worst = {k: 0 for k in exp}
        for i, v in PR.items():
            if i not in B or not is_valid(B[i]): continue
            lad = sorted(int(L) for L, d in (v.get('per_L') or {}).items() if (d.get('n_composer_rivals') or 0) > 0)
            if not lad: continue
            n_items += 1; Rg = []
            for L in lad:
                d = v['per_L'][str(L)]; n = d['n_composer_rivals']
                b = sum(1 for x in d.get('blockers', []) if x.get('kind') == 'composer')
                s = any(x.get('kind') == 'structural' for x in d.get('blockers', []))
                Rg.append((n, b, s)); nr += 1
                if b: nb += 1; shares.append(b / n)
                if b and not s: only += 1
            f = next((j for j, (n, b, s) in enumerate(Rg) if b == 0 and not s), None)
            for k in exp:
                pb = 1.0; ps = None
                for j, (n, b, s) in enumerate(Rg):
                    pp = 0.0 if s else (1.0 if b == 0 else comb(n - b, min(k, n)) / comb(n, min(k, n)))
                    if j == f: ps = pb * pp; break
                    pb *= 1 - pp
                exp[k] += 1 - (pb if ps is None else ps)
                last = f if f is not None else len(Rg)
                worst[k] += any((not s) and b > 0 and (n - b) >= min(k, n) and min(k, n) < n for n, b, s in Rg[:last])
        print(f"  {m}: {n_items} items, {nr} rungs, composed rivals block at {nb} ({only} with no structural blocker), share {st.mean(shares):.0%}")
        print(f"      worst case {worst}   expected {({k: round(v, 1) for k, v in exp.items()})}")


def s_supra():
    head("supra: each examinee in its own supra window (Qwen3-14B)")
    M = measures("Qwen3-14B"); items = sorted(M); acc = accuracy("Qwen3-14B", set(items))
    for e in sorted(acc, key=short):
        it = [i for i in items if i in acc[e]]; t, s = fit([M[i]['Kt'] for i in it], [acc[e][i] for i in it])
        start = t + s * math.log(2); w = [i for i in it if M[i]['Kt'] > start]
        if not w: continue
        k = sum(1 for i in w if acc[e][i] > 0.5); ci = stats.binomtest(k, len(w)).proportion_ci(method='exact')
        print(f"  {short(e):26s} window from {start:4.0f}: {len(w):2d} items, {k:2d} solved on a majority [{ci.low:.0%}, {ci.high:.0%}]")
    hi = [i for i in items if M[i]['Kt'] > 190]
    S = np.array([[1.0 if acc[e].get(i, 0) > 0.5 else 0.0 for e in STRONG] for i in hi])
    bs = [S[RNG.integers(0, len(hi), len(hi))].mean() for _ in range(10000)]
    print(f"  six strongest above 190 bits: {S.mean():.0%} solved, [{np.percentile(bs,2.5):.0%}, {np.percentile(bs,97.5):.0%}] resampling items; "
          f"{int((S.sum(1)==0).sum())} of {len(hi)} items solved by none")


def s_misspecified():
    head("misspecified: shared wrong answers, and results without the flagged items")
    norm = lambda a: re.sub(r'\s+', '', str(a or '').lower().strip().rstrip('.'))
    for m in MACHINES:
        M = measures(m); items = sorted(M); rr = rows_at(m, set(items)); by = collections.defaultdict(list)
        for r in rr: by[r['item']].append(r)
        top = []
        for i in items:
            wrong = collections.Counter(norm(r.get('answer')) for r in by[i] if full(r) == 0 and norm(r.get('answer')))
            if wrong:
                a, n = wrong.most_common(1)[0]; top.append((n / len(by[i]), i, a))
        print(f"  {m}: most shared wrong answers")
        for sh, i, a in sorted(top, reverse=True)[:6]: print(f"     {i}  {sh:4.0%}  {a[:60]!r}")
        acc = accuracy(m, set(items)); y = mean_accuracy(acc, items); keep = [i for i in items if i not in FLAGGED[m]]
        print(f"     Spearman(Kt, accuracy): all {rho([M[i]['Kt'] for i in items],[y[i] for i in items]):+.2f}, "
              f"without {sorted(FLAGGED[m])} {rho([M[i]['Kt'] for i in keep],[y[i] for i in keep]):+.2f}")


SECTIONS = dict(prior=s_prior, ratio=s_ratio, kt_min=s_kt_min, tau=s_tau, upper=s_upper, seed=s_seed,
                budget=s_budget, refusals=s_refusals, judge=s_judge, rivals=s_rivals, supra=s_supra,
                misspecified=s_misspecified)
if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SECTIONS)): SECTIONS[name]()
