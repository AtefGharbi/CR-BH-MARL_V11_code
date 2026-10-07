import json, glob, collections, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
rng = np.random.default_rng(0)
OUT = {}
R = []
for f in glob.glob('res/main_*.json'):
    R += [dict(r, tag='main') for r in json.load(open(f))]
for f in sorted(glob.glob('jobs/*.json')):
    R += json.load(open(f))
by = collections.defaultdict(list)
for r in R:
    by[(r['tag'], r['method'], r['attack'], r['f'], r['N'])].append(r)

M9 = ["non_robust", "median", "trimmed_mean", "krum", "multi_krum", "reputation_only",
      "anomaly_filtering_only", "robust_strategic_only", "crbhmarl"]
ALLM = M9 + ["independent_stores", "oracle"]
A = ["random_fitness", "fitness_inflation", "sign_flip", "colluding", "stealthy", "adaptive"]
FG = [2, 3, 6, 9, 12]
FFINE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LBL = dict(non_robust='Non-robust', median='Median', trimmed_mean='Trimmed mean', krum='Krum',
           multi_krum='Multi-Krum', reputation_only='Reputation only',
           anomaly_filtering_only='Anomaly filtering only', robust_strategic_only='Robust strategic only',
           crbhmarl='CR-BH-MARL', independent_stores='Independent stores', oracle='Oracle reference')
ALBL = dict(random_fitness='A1 Random', fitness_inflation='A2 Inflation', sign_flip='A3 Sign flip',
            colluding='A4 Colluding', stealthy='A5 Stealthy', adaptive='A6-PB Adaptive', adaptive_wb='A6-WB')


def cell(m, a, f, tag='main', N=30):
    if f == 0:
        return by[('main', m, 'none', 0, N)]
    rs = by[(tag, m, a, f, N)]
    if not rs and tag == 'main':
        rs = by[('ref', m, a, f, N)]
    return rs


def mean(rs, k='dsr'):
    return float(np.mean([r[k] for r in rs])) if rs else float('nan')


def rates(rs):
    tp = sum(r['tp'] for r in rs); fn = sum(r['fn'] for r in rs)
    fp = sum(r['fp'] for r in rs); tn = sum(r['tn'] for r in rs)
    rec = tp / (tp + fn) if tp + fn else float('nan')
    fpr = fp / (fp + tn) if fp + tn else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    return rec, fpr, prec


def curve(m, a, fs=FFINE, tag='main'):
    return [mean(cell(m, a, f, tag)) for f in [0] + fs]


# ---------------- clean
OUT['clean'] = {m: mean(cell(m, None, 0)) for m in ALLM}
OUT['clean_loss'] = {m: (float(np.mean([r['loss'] for r in cell(m, None, 0)])),
                         float(np.std([r['loss'] for r in cell(m, None, 0)]))) for m in ALLM}
# ---------------- Table 5 (non-robust) with components
T5 = {}
for a in A:
    T5[a] = {f: dict(dsr=mean(cell('non_robust', a, f)), mit=mean(cell('non_robust', a, f), 'mit'),
                     safe=mean(cell('non_robust', a, f), 'safe'), alloc=mean(cell('non_robust', a, f), 'alloc'))
             for f in FG}
OUT['T5'] = T5
# ---------------- Table 6a/6b
OUT['T6'] = {a: {m: {f: mean(cell(m, a, f)) for f in [0] + FG}
                 for m in ['non_robust', 'reputation_only', 'robust_strategic_only', 'crbhmarl', 'oracle']}
             for a in ['stealthy', 'adaptive', 'fitness_inflation', 'sign_flip']}


# ---------------- Table 7 breakdown with bootstrap CI
def eta_star(curve_vals, fs=[0] + FFINE):
    base = curve_vals[0]; last = 0.0
    for f, v in zip(fs, curve_vals):
        if v >= base - 0.05:
            last = f / 30
        else:
            break
    return last


T7 = {}
for m in ALLM:
    T7[m] = {}
    for a in A:
        cells = [cell(m, a, f) for f in [0] + FFINE]
        mats = np.array([[r['dsr'] for r in sorted(c, key=lambda r: r['seed'])] for c in cells], float)
        est = eta_star(mats.mean(1))
        boots = []
        for _ in range(500):
            idx = rng.integers(0, mats.shape[1], mats.shape[1])
            boots.append(eta_star(mats[:, idx].mean(1)))
        T7[m][a] = (est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
                    bool(mats[-1].mean() >= mats[0].mean() - 0.05))
OUT['T7'] = T7
# ---------------- best baseline
single = ["median", "trimmed_mean", "krum", "multi_krum", "reputation_only", "robust_strategic_only"]
avg = {m: float(np.mean([mean(cell(m, a, f)) for a in A for f in FG])) for m in single + ['crbhmarl', 'non_robust', 'anomaly_filtering_only', 'independent_stores', 'oracle']}
OUT['avgDSR'] = avg
best = max(single, key=lambda m: avg[m]); OUT['best'] = best
# ---------------- Table 8
T8 = {}
for m in ['reputation_only', 'robust_strategic_only', 'independent_stores', 'crbhmarl']:
    rs = [r for a in ['colluding', 'adaptive'] for f in FG for r in cell(m, a, f)]
    rec, fpr, prec = rates(rs)
    dc = float(np.mean([mean(cell(m, 'colluding', f)) for f in FG]))
    da = float(np.mean([mean(cell(m, 'adaptive', f)) for f in FG]))
    T8[m] = dict(dsr_coll=dc, dsr_adapt=da, ai=OUT['clean'][m] - (dc + da) / 2, recall=rec, fpr=fpr, prec=prec,
                 bir=float(np.median([r['bir'] for r in rs])),
                 dsr_coll40=mean(cell(m, 'colluding', 12)), dsr_adapt40=mean(cell(m, 'adaptive', 12)))
OUT['T8'] = T8
# per attack shared vs independent (all attacks) for discussion
OUT['shared_vs_indep'] = {a: (float(np.mean([mean(cell('crbhmarl', a, f)) for f in FG])),
                              float(np.mean([mean(cell('independent_stores', a, f)) for f in FG]))) for a in A}
# detection by attack for crbhmarl at eta=0.3
OUT['det'] = {m: {a: rates(cell(m, a, 9)) for a in A} for m in ['crbhmarl', 'independent_stores', 'reputation_only', 'anomaly_filtering_only', 'robust_strategic_only']}
OUT['bir_tab'] = {m: {a: float(np.median([r['bir'] for r in cell(m, a, 9)])) for a in A} for m in ['non_robust', 'median', 'krum', 'crbhmarl', 'oracle']}


# ---------------- paired statistics
def paired(m1, m2, attacks, fs=FG):
    """seed-paired DSR difference m1-m2 over attacks x fs; bootstrap CI over seeds; exact McNemar"""
    D = np.zeros(30); b = c = 0
    for a in attacks:
        for f in fs:
            x = {r['seed']: r['dsr'] for r in cell(m1, a, f)}; y = {r['seed']: r['dsr'] for r in cell(m2, a, f)}
            for s in range(30):
                D[s] += x[s] - y[s]
                b += x[s] and not y[s]; c += y[s] and not x[s]
    D /= len(attacks) * len(fs)
    boots = [D[rng.integers(0, 30, 30)].mean() for _ in range(5000)]
    p = binomtest(int(b), int(b + c), 0.5).pvalue if b + c else 1.0
    return dict(diff=float(D.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                b=int(b), c=int(c), p=float(p), OR=float((b + 0.5) / (c + 0.5)))


S = {}
S['P1_cr_vs_nonrobust_all'] = paired('crbhmarl', 'non_robust', A)
S['P2_cr_vs_best_A4A6'] = paired('crbhmarl', best, ['colluding', 'adaptive'])
S['P3_shared_vs_indep_A4A6'] = paired('crbhmarl', 'independent_stores', ['colluding', 'adaptive'])
S['cr_vs_best_A4'] = paired('crbhmarl', best, ['colluding'])
S['cr_vs_best_A6'] = paired('crbhmarl', best, ['adaptive'])
S['shared_vs_indep_A4'] = paired('crbhmarl', 'independent_stores', ['colluding'])
S['shared_vs_indep_A6'] = paired('crbhmarl', 'independent_stores', ['adaptive'])
S['H4_cr_vs_nonrobust_A6'] = paired('crbhmarl', 'non_robust', ['adaptive'])
S['shared_vs_indep_all'] = paired('crbhmarl', 'independent_stores', A)
for a in A:
    S['shared_vs_indep_' + a] = paired('crbhmarl', 'independent_stores', [a])
    S['cr_vs_best_' + a] = paired('crbhmarl', best, [a])
    S['cr_vs_median_' + a] = paired('crbhmarl', 'median', [a])
# Holm over primary family
prim = ['P1_cr_vs_nonrobust_all', 'P2_cr_vs_best_A4A6', 'P3_shared_vs_indep_A4A6']
ps = sorted([(S[k]['p'], k) for k in prim]); k_ = len(ps); run = 0
for i, (p, k) in enumerate(ps):
    run = max(run, min(1, p * (k_ - i))); S[k]['p_holm'] = run
OUT['stats'] = S

# H1: GEE binomial trend, non-robust, attack fixed effects, clustered by seed
try:
    import statsmodels.api as sm, statsmodels.formula.api as smf, pandas as pd
    rows = []
    for a in A:
        for f in [0] + FFINE:
            for r in cell('non_robust', a, f):
                rows.append(dict(y=int(r['dsr']), eta=f / 30, attack=a, seed=r['seed']))
    df = pd.DataFrame(rows)
    g = smf.gee('y ~ eta + C(attack)', 'seed', df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    OUT['H1'] = dict(coef=float(g.params['eta']), lo=float(g.conf_int().loc['eta', 0]),
                     hi=float(g.conf_int().loc['eta', 1]), p=float(g.pvalues['eta']))
    # per-attack Spearman-like check
except Exception as e:
    OUT['H1'] = dict(err=str(e))

# ---------------- Table 9 signal ablation (eta=0.30, all six attacks)
def ablrow(tag):
    rs = [r for a in A for r in (cell('crbhmarl', a, 9) if tag == 'main' else by[(tag, 'crbhmarl', a, 9, 30)])]
    rec, fpr, _ = rates(rs)
    clean = cell('crbhmarl', None, 0) if tag == 'main' else by[(tag, 'crbhmarl', 'none', 0, 30)]
    _, fpr0, _ = rates(clean)
    return dict(dsr=mean(rs), rec=rec, fpr=fpr, fpr0=fpr0, per={a: mean(cell('crbhmarl', a, 9) if tag == 'main' else by[(tag, 'crbhmarl', a, 9, 30)]) for a in A})


T9 = {'full': ablrow('main')}
for s in ['fit', 'grad', 'temp', 'nbr', 'strat']:
    T9[s] = ablrow('abl_' + s)
OUT['T9'] = T9
# ---------------- Table 10 non-IID
T10 = {}
for m in ['median', 'krum', 'reputation_only', 'crbhmarl']:
    iid = [r for a in A for r in cell(m, a, 9)]
    nid = [r for a in A for r in by[('noniid', m, a, 9, 30)]]
    T10[m] = dict(iid=mean(iid), noniid=mean(nid), fpr_iid=rates(cell(m, None, 0))[1],
                  fpr_noniid=rates(by[('noniid', m, 'none', 0, 30)])[1],
                  fpr_iid_att=rates(iid)[1], fpr_noniid_att=rates(nid)[1],
                  clean_noniid=mean(by[('noniid', m, 'none', 0, 30)]))
OUT['T10'] = T10
# ---------------- topology
TOP = {}
for d, tag in [(3, 'topo3'), (5, 'main'), (10, 'topo10')]:
    TOP[d] = {}
    for a in ['colluding', 'stealthy', 'fitness_inflation']:
        cs = [by[(tag, 'crbhmarl', 'none', 0, 30)] if tag != 'main' else cell('crbhmarl', None, 0)] + \
             [by[(tag, 'crbhmarl', a, f, 30)] if tag != 'main' else cell('crbhmarl', a, f) for f in FG]
        TOP[d][a] = [mean(c) for c in cs]
        TOP[d][a + '_rf'] = rates([r for c in cs[1:] for r in c])
    c0 = by[(tag, 'crbhmarl', 'none', 0, 30)] if tag != 'main' else cell('crbhmarl', None, 0)
    TOP[d]['deg'] = float(c0[0]['degree']); TOP[d]['fpr0'] = rates(c0)[1]
OUT['TOP'] = TOP
# ---------------- sensitivity
SEN = {}
SA = ['colluding', 'sign_flip', 'stealthy', 'fitness_inflation']
def senrow(tag):
    if tag == 'main':
        rs = [r for a in SA for r in cell('crbhmarl', a, 9)]; c0 = cell('crbhmarl', None, 0)
    else:
        rs = [r for a in SA for r in by[(tag, 'crbhmarl', a, 9, 30)]]; c0 = by[(tag, 'crbhmarl', 'none', 0, 30)]
    rec, fpr, _ = rates(rs)
    return dict(dsr=mean(rs), rec=rec, fpr=fpr, fpr0=rates(c0)[1], clean=mean(c0),
                per={a: mean([r for r in rs if r['attack'] == a]) for a in SA})
SEN['base'] = senrow('main')
for key, vals in [('tau_mult', [1, 2, 4]), ('lam', [0.8, 0.95, 0.99]), ('gamma', [2, 3])]:
    for v in vals:
        SEN[f'{key}={v}'] = senrow(f'sens_{key}_{v}')
OUT['SEN'] = SEN
# ---------------- severity
SEV = {}
for key, vals, a in [('q_A5', [0.5, 0.8, 0.95], 'stealthy'), ('q_A6', [0.5, 1.0, 2.0], 'adaptive'), ('c', [1.0, 3.0, 5.0], 'sign_flip')]:
    for v in vals:
        for m in ['non_robust', 'median', 'reputation_only', 'crbhmarl']:
            mid = {'q_A5': 0.8, 'q_A6': 1.0, 'c': 3.0}[key]
            cs = [cell(m, a, f) if v == mid else by[(f'sev_{key}_{v}', m, a, f, 30)] for f in FG]
            SEV[f'{key}={v}|{m}'] = dict(dsr=[mean(c) for c in cs], bir=[float(np.median([r['bir'] for r in c])) for c in cs],
                                        rec=rates([r for c in cs for r in c])[0])
OUT['SEV'] = SEV
# A6-WB
OUT['WB'] = {m: [mean(by[('wb', m, 'adaptive_wb', f, 30)]) for f in FG] for m in ALLM}
OUT['WB_bir'] = {m: [float(np.median([r['bir'] for r in by[('wb', m, 'adaptive_wb', f, 30)]])) for f in FG] for m in ALLM}
OUT['PB_bir'] = {m: [float(np.median([r['bir'] for r in cell(m, 'adaptive', f)])) for f in FG] for m in ALLM}
# ---------------- scalability
SC = {}
for N in [10, 30, 50, 100]:
    SC[N] = {}
    f = int(np.floor(0.3 * N + 0.5))
    for m in ['non_robust', 'median', 'krum', 'reputation_only', 'crbhmarl']:
        rs = cell(m, 'colluding', f) if N == 30 else by[(f'scale{N}', m, 'colluding', f, N)]
        SC[N][m] = dict(t=float(np.mean([r['t_step'] for r in rs]) * 1e3), dsr_coll=mean(rs))
        if m in ['non_robust', 'reputation_only', 'crbhmarl']:
            rs2 = cell(m, 'fitness_inflation', f) if N == 30 else by[(f'scale{N}', m, 'fitness_inflation', f, N)]
            SC[N][m]['dsr_infl'] = mean(rs2)
OUT['SC'] = SC
# trust trajectories
TR = {}
for m in ['crbhmarl', 'independent_stores']:
    for a in ['colluding', 'sign_flip', 'stealthy', 'adaptive']:
        rs = by[('traj', m, a, 9, 30)]
        H = np.array([r['trace']['rH'] for r in rs]); B = np.array([r['trace']['rB'] for r in rs])
        TR[f'{m}|{a}'] = dict(rH=H.mean(0).tolist(), rB=B.mean(0).tolist(), rHsd=H.std(0).tolist(), rBsd=B.std(0).tolist())
OUT['TR'] = TR
json.dump(OUT, open('results.json', 'w'), indent=1, default=float)

# =========================== FIGURES ===========================
os.makedirs('figs', exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False})
X = np.array([0] + FFINE) / 30 * 100
COL = dict(non_robust='#7f7f7f', median='#1f77b4', reputation_only='#2ca02c', robust_strategic_only='#9467bd',
           crbhmarl='#d62728', independent_stores='#ff7f0e', oracle='black', krum='#17becf',
           anomaly_filtering_only='#8c564b', trimmed_mean='#bcbd22', multi_krum='#e377c2')
MK = dict(random_fitness='o', fitness_inflation='s', sign_flip='^', colluding='D', stealthy='v', adaptive='P')


def ci(p, n=30):
    return 1.96 * np.sqrt(np.clip(p * (1 - p), 0, None) / n)


# Fig 7
fig, ax = plt.subplots(figsize=(6, 3.1))
for a in A:
    y = np.array(curve('non_robust', a)); ax.plot(X, y * 100, marker=MK[a], ms=3.5, lw=1.3, label=ALBL[a])
ax.set_xlabel(r'Realized Byzantine fraction $\eta_{\rm real}$ (%)'); ax.set_ylabel('DSR (%)'); ax.set_ylim(-3, 103)
ax.legend(ncol=3, fontsize=7.5, loc='lower left'); fig.tight_layout(); fig.savefig('figs/fig7.png', dpi=220); plt.close()

# Fig 8
fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.3), sharex=True, sharey=True)
for ax, a in zip(axs.flat, A):
    for m in ['non_robust', 'median', best, 'crbhmarl', 'oracle']:
        y = np.array(curve(m, a))
        ax.plot(X, y * 100, color=COL[m], lw=1.6 if m == 'crbhmarl' else 1.1, ls='--' if m == 'oracle' else '-',
                marker='o' if m == 'crbhmarl' else None, ms=2.5, label=LBL[m] + (' (best single-layer)' if m == best else ''))
        if m == 'crbhmarl':
            ax.fill_between(X, (y - ci(y)) * 100, (y + ci(y)) * 100, color=COL[m], alpha=0.12, lw=0)
    ax.set_title(ALBL[a], fontsize=8.5); ax.set_ylim(-3, 105)
for ax in axs[1]: ax.set_xlabel(r'$\eta_{\rm real}$ (%)')
for ax in axs[:, 0]: ax.set_ylabel('DSR (%)')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figs/fig8.png', dpi=220); plt.close()

# Fig 9 shared vs independent
fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.3), sharey=True)
for ax, a in zip(axs, ['colluding', 'sign_flip', 'adaptive', 'stealthy']):
    for m in ['independent_stores', 'crbhmarl', 'reputation_only', 'robust_strategic_only']:
        y = np.array(curve(m, a))
        ax.plot(X, y * 100, color=COL[m], lw=1.6 if m in ('crbhmarl', 'independent_stores') else 0.9,
                ls='-' if m in ('crbhmarl', 'independent_stores') else ':',
                label={'crbhmarl': 'Shared store (CR-BH-MARL)', 'independent_stores': 'Independent stores',
                       'reputation_only': 'Coordination only', 'robust_strategic_only': 'Strategic only'}[m])
        if m in ('crbhmarl', 'independent_stores'):
            ax.fill_between(X, (y - ci(y)) * 100, (y + ci(y)) * 100, color=COL[m], alpha=0.12, lw=0)
    ax.set_title(ALBL[a], fontsize=8.5); ax.set_xlabel(r'$\eta_{\rm real}$ (%)')
axs[0].set_ylabel('DSR (%)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=7.2)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig('figs/fig9.png', dpi=220); plt.close()

# Fig 10 trust trajectories
fig, axs = plt.subplots(2, 4, figsize=(6.8, 3.6), sharex=True, sharey=True)
for j, a in enumerate(['colluding', 'sign_flip', 'stealthy', 'adaptive']):
    for i, m in enumerate(['independent_stores', 'crbhmarl']):
        ax = axs[i, j]; d = TR[f'{m}|{a}']; t = np.arange(1, len(d['rH']) + 1)
        for k, c, lab in [('rH', '#1f77b4', r'$\bar r_{\mathcal{H}}(t)$'), ('rB', '#d62728', r'$\bar r_{\mathcal{B}}(t)$')]:
            y = np.array(d[k]); s = np.array(d[k + 'sd'])
            ax.plot(t, y, color=c, lw=1.3, label=lab); ax.fill_between(t, y - s, y + s, color=c, alpha=0.12, lw=0)
        ax.axhline(0.65, color='k', ls=':', lw=0.8); ax.axhline(0.72, color='gray', ls=':', lw=0.6)
        if i == 0: ax.set_title(ALBL[a], fontsize=8.5)
        if i == 1: ax.set_xlabel('Coordination round $t$')
    axs[0, 0].set_ylabel('Independent\nmean trust'); axs[1, 0].set_ylabel('Shared (CR-BH-MARL)\nmean trust')
axs[0, 0].set_ylim(0.2, 1.02)
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig('figs/fig10.png', dpi=220); plt.close()

# Fig 11 adaptive attacks: DSR & BIR
XG = np.array(FG) / 30 * 100
fig, axs = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
cases = [('stealthy', 'A5 Stealthy (q=0.8)'), ('adaptive', 'A6-PB (q=1)'), ('adaptive_wb', 'A6-WB oracle attacker')]
for j, (a, ttl) in enumerate(cases):
    for m in ['non_robust', 'median', 'reputation_only', 'crbhmarl', 'oracle']:
        if a == 'adaptive_wb':
            y = OUT['WB'][m]; b = OUT['WB_bir'][m]
        else:
            y = [mean(cell(m, a, f)) for f in FG]; b = [float(np.median([r['bir'] for r in cell(m, a, f)])) for f in FG]
        axs[0, j].plot(XG, np.array(y) * 100, color=COL[m], marker='o', ms=2.5, lw=1.2, ls='--' if m == 'oracle' else '-', label=LBL[m])
        axs[1, j].plot(XG, b, color=COL[m], marker='o', ms=2.5, lw=1.2, ls='--' if m == 'oracle' else '-')
    axs[0, j].set_title(ttl, fontsize=8.5); axs[1, j].set_xlabel(r'$\eta_{\rm real}$ (%)'); axs[0, j].set_ylim(-3, 105)
axs[0, 0].set_ylabel('DSR (%)'); axs[1, 0].set_ylabel('BIR (median, final round)')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, fontsize=7.2)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig('figs/fig11.png', dpi=220); plt.close()

# Fig 12 scalability
Ns = [10, 30, 50, 100]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3))
for m in ['non_robust', 'median', 'krum', 'reputation_only', 'crbhmarl']:
    axs[0].plot(Ns, [SC[N][m]['t'] for N in Ns], marker='o', ms=3, color=COL[m], label=LBL[m])
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_xlabel('$N$'); axs[0].set_ylabel('Defense compute per round (ms)')
axs[0].set_xticks(Ns); axs[0].set_xticklabels(Ns); axs[0].legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
for m in ['non_robust', 'reputation_only', 'crbhmarl']:
    axs[1].plot(Ns, [SC[N][m]['dsr_coll'] * 100 for N in Ns], marker='D', ms=3, color=COL[m], label=LBL[m] + ', A4')
    axs[1].plot(Ns, [SC[N][m]['dsr_infl'] * 100 for N in Ns], marker='s', ms=3, ls=':', color=COL[m], label=LBL[m] + ', A2')
axs[1].set_xscale('log'); axs[1].set_xticks(Ns); axs[1].set_xticklabels(Ns); axs[1].set_xlabel('$N$')
axs[1].set_ylabel(r'DSR (%), $\eta=0.30$'); axs[1].set_ylim(-3, 105); axs[1].legend(fontsize=6.0, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout(); fig.savefig('figs/fig12.png', dpi=220); plt.close()

# Fig 13 topology heat maps
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.0))
for ax, a in zip(axs, ['colluding', 'stealthy', 'fitness_inflation']):
    Z = np.array([TOP[d][a] for d in [3, 5, 10]]) * 100
    im = ax.imshow(Z, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto', origin='lower')
    ax.set_xticks(range(6)); ax.set_xticklabels(['0', '6.7', '10', '20', '30', '40'], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{TOP[d]['deg']:.1f}" for d in [3, 5, 10]], fontsize=7)
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f'{Z[i, j]:.0f}', ha='center', va='center', fontsize=6.5)
    ax.set_title(ALBL[a], fontsize=8.5); ax.set_xlabel(r'$\eta_{\rm real}$ (%)'); ax.grid(False)
axs[0].set_ylabel(r'realized $\bar d$')
fig.tight_layout(); fig.savefig('figs/fig13.png', dpi=220); plt.close()
print('ok best=', best)
