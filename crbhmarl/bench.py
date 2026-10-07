"""
bench.py -- CR-BH-MARL benchmark runner on the IEEE 33-bus feeder
(ieee33_fast.IEEE33FastEnv). Implements the nine deployable methods of
Section 6.7, the independent-store ablation (Section 6.10), the
Oracle-Honest Cross-Layer Reference, attacks A1-A6 (A6-PB and A6-WB,
Section 6.4), the three-component DSR (Section 6.3) and the detection /
BIR metrics of Section 6.16.

Uses reputation.py (psi, cos_eps, TrustStore, AgentSignalHistory) and
robust_agg.py unchanged.
"""
from __future__ import annotations
import time
import numpy as np
from .reputation import psi, TrustStore, AgentSignalHistory
from .robust_agg import naive_mean, coordinate_wise_median, trimmed_mean, krum, multi_krum
from .ieee33_fast import IEEE33FastEnv

COORD = ("fit", "temp", "nbr")
STRAT = ("grad", "strat")
ALL = ("fit", "grad", "temp", "nbr", "strat")
METHODS = ["non_robust", "median", "trimmed_mean", "krum", "multi_krum",
           "reputation_only", "anomaly_filtering_only", "robust_strategic_only",
           "crbhmarl", "independent_stores", "oracle"]
ATTACKS = ["random_fitness", "fitness_inflation", "sign_flip", "colluding",
           "stealthy", "adaptive"]

DEFAULT = dict(lam=0.9, r_min=0.65, r_re=0.72, gamma=1.0, tau_mult=3.0, W=10,
               M=5, k_frac=1 / 3, eta_bar=0.30, degree=5, sigma_B_mult=2.0,
               c=3.0, q_A5=0.8, q_A6=1.0, h_max=0.0, n_steps=40, lr=0.5,
               drop_signals=(), non_iid=False)


def realized_f(eta, n):
    return 0 if eta <= 0 else max(1, int(np.floor(eta * n + 0.5)))


def neighbour_graph(env, degree):
    """Symmetric k-nearest graph on electrical hop distance; k chosen so
    that the realised average degree is closest to `degree`."""
    H = env.F.hops[np.ix_(env.agent_bus, env.agent_bus)].astype(float)
    n = len(H)
    H = H + 1e-3 * np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    np.fill_diagonal(H, np.inf)
    best = None
    for k in range(1, min(n - 1, 3 * degree) + 1):
        A = np.zeros((n, n), bool)
        idx = np.argsort(H, axis=1)[:, :k]
        A[np.arange(n)[:, None], idx] = True
        A = A | A.T
        d = A.sum(1).mean()
        if best is None or abs(d - degree) < abs(best[1] - degree):
            best = (A, d)
    A = best[0]
    return [np.flatnonzero(A[i]) for i in range(n)], best[1]


def _fuse(comp, active, drop):
    act = [k for k in active if k not in drop]
    if not act:
        return np.zeros(len(next(iter(comp.values()))))
    return np.mean([comp[k] for k in act], axis=0)  # uniform weights over active signals


def _cos_rows(G, ref, eps=1e-8):
    return (G @ ref) / ((np.linalg.norm(G, axis=1) + eps) * (np.linalg.norm(ref) + eps))


def run_episode(method, attack, eta, n_agents, seed, cfg=None, record=False, attack_seed=None):
    cfg = {**DEFAULT, **(cfg or {})}
    env = IEEE33FastEnv(n_agents, seed=seed, lr=cfg["lr"], n_steps=cfg["n_steps"],
                        non_iid=cfg["non_iid"])
    rng = np.random.default_rng(10_000 + seed if attack_seed is None else attack_seed)
    N = n_agents
    f = realized_f(eta, N)
    byz = np.sort(rng.choice(N, size=f, replace=False)) if f else np.array([], int)
    isB = np.zeros(N, bool); isB[byz] = True
    nbrs, deg = neighbour_graph(env, cfg["degree"])
    k_sel = max(1, int(round(cfg["k_frac"] * N)))
    f_max = int(np.floor(cfg["eta_bar"] * N))
    if method in ("krum", "multi_krum") and N < 2 * f_max + 3:
        return None

    rep = method in ("reputation_only", "anomaly_filtering_only", "robust_strategic_only",
                     "crbhmarl", "independent_stores")
    lam = 0.0 if method == "anomaly_filtering_only" else cfg["lam"]
    mk = lambda: TrustStore(N, lam=lam, r_min=cfg["r_min"], r_reentry=cfg["r_re"], gamma=cfg["gamma"])
    store = mk(); store2 = mk() if method == "independent_stores" else None
    coord_prot = method in ("reputation_only", "anomaly_filtering_only", "crbhmarl", "independent_stores")
    strat_prot = method in ("robust_strategic_only", "anomaly_filtering_only", "crbhmarl", "independent_stores")
    drop = set(cfg["drop_signals"])
    if method == "anomaly_filtering_only":
        drop.add("temp")
    hist = [AgentSignalHistory(window=cfg["W"]) for _ in range(N)]
    strat_cache = np.zeros(N)
    g_prev = None; tau_prev = None
    u_perp = rng.normal(size=N)
    trace = {"rH": [], "rB": [], "aH": [], "aB": []} if record else None
    t_step = 0.0

    for t in range(cfg["n_steps"]):
        h = env.all_fitness(); G = env.all_gradients()
        hh = h.copy(); GG = G.copy()
        # local deviation from neighbourhood median, robust scale, detection threshold
        med_nb = np.array([np.median(hh[nbrs[i]]) for i in range(N)])
        dev_true = h - med_nb
        sigma_h = 1.4826 * np.median(np.abs(dev_true)) + 1e-6
        tau_att = tau_prev if tau_prev is not None else cfg["tau_mult"] * sigma_h
        # ---------------- attacks (Byzantine reports) ----------------
        if f:
            if attack == "random_fitness":
                hh[byz] = h[byz] + rng.normal(0, cfg["sigma_B_mult"] * sigma_h, f)
            elif attack == "fitness_inflation":
                hh[byz] = cfg["h_max"]
            elif attack == "sign_flip":
                GG[byz] = -cfg["c"] * G[byz]
            elif attack == "colluding":
                hh[byz] = cfg["h_max"]; GG[byz] = -cfg["c"] * G[byz]
            elif attack == "stealthy":
                hh[byz] = h[byz] + cfg["q_A5"] * tau_att * np.sign(cfg["h_max"] - h[byz])
            elif attack in ("adaptive", "adaptive_wb"):
                if attack == "adaptive_wb":
                    c0 = G[~isB].mean(0)
                    eps = cfg["q_A6"] * np.median(np.linalg.norm(G[~isB] - c0, axis=1))
                else:  # A6-PB: previous realised update + colluders' own dispersion
                    c0 = g_prev if g_prev is not None else G[byz].mean(0)
                    eps = cfg["q_A6"] * np.median(np.linalg.norm(G[byz] - G[byz].mean(0), axis=1))
                nc = np.linalg.norm(c0) + 1e-12; ch = c0 / nc
                up = u_perp - (u_perp @ ch) * ch; up /= np.linalg.norm(up) + 1e-12
                if eps < nc:   # tangent point of the eps-ball: minimum cosine to c0
                    x = c0 * (1 - eps ** 2 / nc ** 2) + eps * np.sqrt(1 - eps ** 2 / nc ** 2) * up
                else:          # ball contains the origin: point straight against c0
                    x = -(eps - nc) * ch
                GG[byz] = x[None, :]
            else:
                raise ValueError(attack)
        t0 = time.perf_counter()
        # ---------------- anomaly signals (pre-aggregation) ----------------
        comp = {}
        if rep:
            med_r = np.array([np.median(hh[nbrs[i]]) for i in range(N)])
            dev = hh - med_r
            sig_r = 1.4826 * np.median(np.abs(dev)) + 1e-6
            tau = cfg["tau_mult"] * sig_r
            comp["fit"] = psi(np.abs(dev) / tau)
            comp["nbr"] = np.array([np.mean(psi(np.abs(hh[i] - hh[nbrs[i]]) / tau)) for i in range(N)])
            for i in range(N):
                hist[i].push(dev[i] / tau)
            comp["temp"] = np.array([hist[i].temporal_score() for i in range(N)])
            cwm = coordinate_wise_median(GG)
            comp["grad"] = psi(1 - _cos_rows(GG, cwm))
            tau_prev = tau
        else:
            tau_prev = cfg["tau_mult"] * sigma_h
        # ---------------- strategic aggregation ----------------
        if method == "independent_stores":
            u = store2.strategic_weights()
        elif strat_prot:
            u = store.strategic_weights()
        else:
            u = np.ones(N)
        keep = u > 0
        X = (GG * u[:, None])[keep] if keep.any() else GG
        if method in ("non_robust", "reputation_only"):
            g_agg = naive_mean(X)
        elif method in ("median", "robust_strategic_only", "anomaly_filtering_only", "crbhmarl", "independent_stores"):
            g_agg = coordinate_wise_median(X)
        elif method == "trimmed_mean":
            g_agg = trimmed_mean(X, beta=cfg["eta_bar"])
        elif method == "krum":
            g_agg = krum(X, f=f_max)
        elif method == "multi_krum":
            g_agg = multi_krum(X, f=f_max)
        elif method == "oracle":
            g_agg = GG[~isB].mean(0)
        else:
            raise ValueError(method)
        # ---------------- post-aggregation strategic signal + ONE trust update ----------------
        if rep:
            if t % cfg["M"] == 0:
                strat_cache = psi(1 - _cos_rows(GG, g_agg))
            comp["strat"] = strat_cache
            if method == "independent_stores":
                store.update(_fuse(comp, COORD, drop)); store2.update(_fuse(comp, STRAT, drop))
            else:
                active = {"reputation_only": COORD, "robust_strategic_only": STRAT}.get(method, ALL)
                a = _fuse(comp, active, drop)
                store.update(a)
                if record:
                    trace["aH"].append(a[~isB].mean()); trace["aB"].append(a[isB].mean() if f else np.nan)
        t_step += time.perf_counter() - t0
        env.apply_aggregate_gradient(g_agg)
        g_prev = g_agg
        if record:
            r = store.r if method != "independent_stores" else 0.5 * (store.r + store2.r)
            trace["rH"].append(r[~isB].mean()); trace["rB"].append(r[isB].mean() if f else np.nan)

    # ---------------- coordination layer: k-WTA confrontation alliance ----------------
    hplus = hh + 0.10  # shift to a non-negative fitness scale (|V-1| <= 0.10 pu)
    if method == "oracle":
        score = np.where(isB, -np.inf, hplus)
    elif coord_prot:
        st = store
        score = np.where(st.excluded, -np.inf, st.r * hplus)
    else:
        score = hplus
    K = np.argsort(-score)[:k_sel]
    y_alloc = bool(isB[K].sum() < k_sel / 2)
    y_mit = env.timely_mitigation(); y_safe = env.physical_safety_held()
    if method == "independent_stores":
        excl = store.excluded | store2.excluded
    elif rep:
        excl = store.excluded
    else:
        excl = np.zeros(N, bool)
    gH = G[~isB].mean(0)
    out = dict(method=method, attack=attack, eta=eta, f=f, N=N, seed=seed,
               mit=y_mit, safe=y_safe, alloc=y_alloc, dsr=y_mit and y_safe and y_alloc,
               tp=int((excl & isB).sum()), fp=int((excl & ~isB).sum()),
               fn=int((~excl & isB).sum()), tn=int((~excl & ~isB).sum()),
               bir=float(np.linalg.norm(g_agg - gH) / (np.linalg.norm(gH) + 1e-8)),
               loss=env.loss(), t_step=t_step / cfg["n_steps"], degree=deg)
    if record:
        out["trace"] = trace
    return out
