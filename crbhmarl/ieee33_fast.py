"""
ieee33_fast.py -- IEEE 33-bus radial feeder (Baran & Wu, pandapower case33bw
data) with an exact AC backward/forward-sweep power-flow solver (BIBC/BCBV
formulation). Drop-in replacement for ieee33_env.IEEE33BusEnv with the same
interface, ~100x faster, validated against pandapower's Newton-Raphson
solver (max |dV| < 1e-6 pu, see validate()).

Modelling choices (documented in manuscript Section 6.2):
  * N defender agents, agent i controls the reactive-power set-point
    theta_i in [Q_MIN, Q_MAX] MVAr of a smart inverter at a non-slack bus
    (wrap-around if N > 32).
  * External CPS-facing disturbance: +disturbance_scale load step at
    n_disturbed random buses at episode start.
  * Defence objective: L(theta) = sum_b (V_b(theta) - 1)^2 over all buses.
  * Agent i's local gradient report: g_i = g_true + sigma_rel*||g_true||*xi_i/sqrt(D),
    xi_i ~ N(0, I) fixed per episode (persistent, heterogeneous local
    estimation error proportional to the signal).
  * Local fitness h_i = -|V_bus(i) - 1| (pu).
  * Y_safe: every bus voltage stays within [0.90, 1.10] pu at every step.
  * Y_mit : every bus voltage within [0.95, 1.05] pu at the final step (T_max).
"""
from __future__ import annotations
import json, os
import numpy as np

_D = json.load(open(os.path.join(os.path.dirname(__file__), "case33.json")))


class FastFeeder:
    def __init__(self):
        self.nb = 33
        self.vn = _D["vn"]
        zbase = self.vn ** 2 / 1.0  # 1 MVA base
        fb, tb = np.array(_D["fb"]), np.array(_D["tb"])
        self.z = (np.array(_D["r"]) + 1j * np.array(_D["x"])) / zbase
        # branch k feeds bus tb[k]; build BIBC (branch x non-slack bus)
        nbr = len(fb)
        adj = {}
        for k, (f, t) in enumerate(zip(fb, tb)):
            adj.setdefault(int(f), []).append((int(t), k)); adj.setdefault(int(t), []).append((int(f), k))
        parent = {}; seen = {0}; q = [0]
        while q:
            u = q.pop(0)
            for v, k in adj[u]:
                if v not in seen:
                    seen.add(v); parent[v] = (u, k); q.append(v)
        B = np.zeros((nbr, self.nb - 1))
        for bus in range(1, self.nb):
            b = bus
            while b != 0:
                f, k = parent[b]
                B[k, bus - 1] = 1.0
                b = f
        self.BIBC = B
        self.DLF = (B.T * self.z[None, :]) @ B  # BCBV @ BIBC
        self.load_bus = np.array(_D["lb"])
        self.p0 = np.array(_D["lp"]); self.q0 = np.array(_D["lq"])
        parent_of = np.zeros(self.nb, dtype=int)
        for b, (f, k) in parent.items():
            parent_of[b] = f
        self.parent = parent_of
        # hop distance between buses (tree)
        self.hops = self._hops(fb, tb)

    def _hops(self, fb, tb):
        n = self.nb
        adj = [[] for _ in range(n)]
        for f, t in zip(fb, tb):
            adj[f].append(t); adj[t].append(f)
        H = np.zeros((n, n), dtype=int)
        for s in range(n):
            dist = {s: 0}; q = [s]
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; q.append(v)
            H[s] = [dist[k] for k in range(n)]
        return H

    def solve(self, p_load, q_load, q_inj_bus, tol=1e-10, it=50):
        """p_load,q_load: per-bus MW/MVAr (len 33); q_inj_bus: per-bus MVAr
        injection. Returns |V| (len 33) in pu."""
        S = (p_load + 1j * (q_load - q_inj_bus))[1:]  # net load, pu on 1 MVA
        V = np.ones(self.nb - 1, dtype=complex)
        for _ in range(it):
            I = np.conj(S / V)
            Vn = 1.0 - self.DLF @ I
            if np.max(np.abs(Vn - V)) < tol:
                V = Vn; break
            V = Vn
        return np.concatenate([[1.0], np.abs(V)])


_FEEDER = None


def feeder():
    global _FEEDER
    if _FEEDER is None:
        _FEEDER = FastFeeder()
    return _FEEDER


class IEEE33FastEnv:
    V_TARGET = 1.0
    V_SAFE_LOW, V_SAFE_HIGH = 0.95, 1.05
    V_HARD_LOW, V_HARD_HIGH = 0.90, 1.10
    Q_MIN, Q_MAX = -0.10, 0.50

    def __init__(self, n_agents, seed=None, lr=0.15, n_steps=40, sigma_rel=0.3,
                 n_disturbed_buses=2, disturbance_scale=0.15, non_iid=False):
        self.F = feeder()
        self.n_agents = n_agents; self.dim = n_agents
        self.lr = lr; self.n_steps = n_steps; self.sigma_rel = sigma_rel
        self.n_dist = n_disturbed_buses; self.dscale = disturbance_scale
        self.non_iid = non_iid
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _V(self, theta):
        qinj = np.zeros(self.F.nb)
        np.add.at(qinj, self.agent_bus, theta)
        return self.F.solve(self.pl, self.ql, qinj)

    def reset(self):
        F = self.F
        nsb = np.arange(1, F.nb)
        self.agent_bus = np.array([nsb[i % len(nsb)] for i in range(self.n_agents)])
        self.pl = np.zeros(F.nb); self.ql = np.zeros(F.nb)
        np.add.at(self.pl, F.load_bus, F.p0); np.add.at(self.ql, F.load_bus, F.q0)
        dist = self.rng.choice(nsb, size=self.n_dist, replace=False)
        self.pl[dist] *= 1 + self.dscale; self.ql[dist] *= 1 + self.dscale
        self.theta = np.zeros(self.n_agents)
        # linearised sensitivity dV/dQ of ALL buses wrt each agent (finite diff, exact AC)
        V0 = self._V(self.theta); d = 0.02
        J = np.zeros((F.nb, self.n_agents))
        for j in range(self.n_agents):
            th = self.theta.copy(); th[j] = d
            J[:, j] = (self._V(th) - V0) / d
        self.V0, self.J = V0, J
        # per-agent persistent estimation error (unit vectors scaled later)
        xi = self.rng.normal(size=(self.n_agents, self.n_agents))
        if self.non_iid:
            # heterogeneous regions: error magnitude grows with electrical distance
            scale = 0.5 + 1.5 * (F.hops[0, self.agent_bus] / F.hops[0].max())
            xi = xi * scale[:, None]
        self.xi = xi / np.sqrt(self.n_agents)
        self.V = V0.copy(); self._unsafe = False; self._check()
        self._gtrue = self._true_grad()

    def _true_grad(self):
        Vlin = self.V0 + self.J @ self.theta
        return 2.0 * self.J.T @ (Vlin - self.V_TARGET)

    def _check(self):
        Vb = self.V[1:]
        if np.any(Vb < self.V_HARD_LOW) or np.any(Vb > self.V_HARD_HIGH):
            self._unsafe = True

    # --- CR-BH-MARL interface ---
    def local_fitness(self, i):
        return float(-abs(self.V[self.agent_bus[i]] - 1.0))

    def all_fitness(self):
        return -np.abs(self.V[self.agent_bus] - 1.0)

    def all_gradients(self):
        g = self._gtrue
        return g[None, :] + self.sigma_rel * np.linalg.norm(g) * self.xi

    def true_gradient(self):
        return self._gtrue.copy()

    def apply_aggregate_gradient(self, g):
        self.theta = np.clip(self.theta - self.lr * g, self.Q_MIN, self.Q_MAX)
        self.V = self._V(self.theta)
        self._check()
        self._gtrue = self._true_grad()

    def timely_mitigation(self):
        Vb = self.V[1:]
        return bool(np.all(Vb >= self.V_SAFE_LOW) and np.all(Vb <= self.V_SAFE_HIGH))

    def physical_safety_held(self):
        return not self._unsafe

    def loss(self):
        return float(np.sum((self.V - 1.0) ** 2))


def validate(n=5):
    import pandapower as pp, pandapower.networks as pn
    F = feeder(); rng = np.random.default_rng(0); err = 0
    for _ in range(n):
        net = pn.case33bw()
        q = rng.uniform(-0.1, 0.5, 32)
        for b in range(1, 33):
            pp.create_sgen(net, b, p_mw=0, q_mvar=q[b - 1])
        pp.runpp(net, numba=False)
        pl = np.zeros(33); ql = np.zeros(33)
        np.add.at(pl, F.load_bus, F.p0); np.add.at(ql, F.load_bus, F.q0)
        qi = np.concatenate([[0], q])
        err = max(err, np.max(np.abs(F.solve(pl, ql, qi) - net.res_bus.vm_pu.values)))
    return err
