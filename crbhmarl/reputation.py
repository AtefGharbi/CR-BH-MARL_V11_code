"""
reputation.py — Cross-Layer Reputation-Aware Byzantine-Robust mechanism.
Implements Section 5.1 (sign convention), 5.2 (six anomaly signals),
5.3 (robust k-WTA reweighting), 5.4 (strategic-layer reweighting), and
Algorithms 1-2 (Section 5.7).

CRITICAL: this file implements the *corrected* V4/V5 sign convention.
a_i(t) in [0,1] is an ANOMALY score (higher = more suspicious).
r_i(t) in [0,1] is the TRUST/reputation state (higher = more trusted),
defined as r_i(t) = lambda*r_i(t-1) + (1-lambda)*(1 - a_i(t)).
Getting this backwards (as the V2 draft did) silently rewards Byzantine
agents with higher trust -- see the V3/V4 revision notes. If you modify
this file, re-derive the sign convention from Proposition 2 (Section 5.6)
before changing anything here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def psi(x: np.ndarray | float, kappa: float = 1.0) -> np.ndarray | float:
    """Saturating normalization mapping raw deviations into [0,1].
    Section 5.2: psi(x) = 1 - exp(-x/kappa). Every anomaly component MUST
    be wrapped in this (or another [0,1]-valued function) before fusion --
    see the V4 correction to a_i^temp, which was not bounded in an earlier
    draft and could push the fused anomaly score above 1.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-np.clip(x, 0, None) / kappa)


def cos_eps(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """Epsilon-regularized cosine similarity (Section 5.2), avoiding
    instability when either vector has near-zero norm."""
    return float(np.dot(x, y) / ((np.linalg.norm(x) + eps) * (np.linalg.norm(y) + eps)))


@dataclass
class AgentSignalHistory:
    """Rolling state needed to compute the temporal-consistency signal
    (Section 5.2) for one agent."""
    window: int = 20
    deltas: list = field(default_factory=list)

    def push(self, delta: float) -> None:
        self.deltas.append(delta)
        if len(self.deltas) > self.window:
            self.deltas.pop(0)

    def temporal_score(self, alpha1: float = 1.0, alpha2: float = 1.0, kappa: float = 1.0) -> float:
        if not self.deltas:
            return 0.0
        arr = np.array(self.deltas)
        raw = alpha1 * abs(arr.mean()) + alpha2 * arr.var()
        return float(psi(raw, kappa))


class SixSignalAnomalyFusion:
    """Implements a_i(t) = sum_k w_k * a_i^(k)(t), the six signals of
    Section 5.2: fitness deviation, gradient-direction inconsistency,
    temporal consistency, neighborhood disagreement, strategic-layer
    anomaly, and observation consistency.

    weights: dict with keys 'fit','grad','temp','nbr','strat','obs'
             summing to 1.0. Not tuned here -- see Section 6.12/6.9 in the
             manuscript for the recommended sensitivity sweep. Defaults to
             uniform weighting as a neutral starting point.
    """

    SIGNAL_NAMES = ("fit", "grad", "temp", "nbr", "strat", "obs")

    def __init__(self, weights: dict[str, float] | None = None, kappa: float = 1.0,
                 alpha1: float = 1.0, alpha2: float = 1.0, use_observation_signal: bool = True):
        if weights is None:
            weights = {k: 1.0 / len(self.SIGNAL_NAMES) for k in self.SIGNAL_NAMES}
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "signal weights must sum to 1"
        self.weights = weights
        self.kappa = kappa
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.use_observation_signal = use_observation_signal

    def fitness_signal(self, h_hat_i: float, neighbor_h_hats: np.ndarray) -> float:
        """a_i^fit: Section 5.2, eq. for fitness deviation."""
        if len(neighbor_h_hats) == 0:
            return 0.0
        med = np.median(neighbor_h_hats)
        return float(psi(abs(h_hat_i - med), self.kappa))

    def gradient_signal(self, g_hat_i: np.ndarray, cwm_direction: np.ndarray) -> float:
        """a_i^grad: Section 5.2, eps-regularized cosine vs. the
        coordinate-wise median *direction* of all reported gradients
        (computed pre-aggregation, to avoid circularity)."""
        return float(psi(1.0 - cos_eps(g_hat_i, cwm_direction), self.kappa))

    def temporal_signal(self, history: AgentSignalHistory) -> float:
        """a_i^temp: Section 5.2, corrected V4 form (wrapped in psi)."""
        return history.temporal_score(self.alpha1, self.alpha2, self.kappa)

    def neighborhood_signal(self, h_hat_i: float, neighbor_h_hats: np.ndarray,
                             sigma_ij: float = 1.0, eps: float = 1e-8) -> float:
        """a_i^nbr: Section 5.2, continuous disagreement measure."""
        if len(neighbor_h_hats) == 0:
            return 0.0
        diffs = np.abs(h_hat_i - neighbor_h_hats) / (sigma_ij + eps)
        return float(np.mean(psi(diffs, self.kappa)))

    def strategic_signal(self, g_i: np.ndarray, g_tilde: np.ndarray) -> float:
        """a_i^strat: Section 5.2, corrected V4 directional form --
        penalizes gradients pointing away from the current robust
        aggregate direction, NOT raw magnitude (the original absolute-value
        formulation would unfairly penalize large-but-aligned honest
        gradients -- see the V4 revision note)."""
        return float(psi(1.0 - cos_eps(g_i, g_tilde), self.kappa))

    def observation_signal(self, o_hat_i: np.ndarray | None, o_pred_i: np.ndarray | None) -> float:
        """a_i^obs: Section 5.2. If no informative consistency predictor is
        available for your environment, explicitly disable this signal via
        use_observation_signal=False rather than silently passing zeros --
        see the Section 5.2 note about scoping this out as a limitation."""
        if not self.use_observation_signal or o_hat_i is None or o_pred_i is None:
            return 0.0
        return float(psi(np.linalg.norm(o_hat_i - o_pred_i), self.kappa))

    def fuse(self, components: dict[str, float]) -> float:
        """a_i(t) = sum_k w_k a_i^(k)(t). Returns a value guaranteed in
        [0,1] since every component and every weight is in [0,1] and the
        weights sum to 1."""
        a = sum(self.weights[k] * components.get(k, 0.0) for k in self.SIGNAL_NAMES)
        return float(np.clip(a, 0.0, 1.0))


class TrustStore:
    """The shared cross-layer reputation store (Section 5.7, Algorithms 1-2).
    This is the paper's central novelty: a SINGLE r_i(t) per agent, updated
    from signals computed at both the coordination layer (fitness-based)
    and the strategic layer (gradient-based), and consumed by both.

    shared=True  -> CR-BH-MARL (the proposed method)
    shared=False -> "independent reputation stores" ablation (Section 6.7);
                     use two separate TrustStore instances instead, one fed
                     only coordination-layer signals and one fed only
                     strategic-layer signals, and do NOT let them read each
                     other's state.
    """

    def __init__(self, n_agents: int, lam: float = 0.9, r_min: float = 0.2,
                 r_reentry: float = 0.4, gamma: float = 1.0, r0: float = 1.0):
        self.n_agents = n_agents
        self.lam = lam
        self.r_min = r_min
        self.r_reentry = r_reentry
        self.gamma = gamma
        self.r = np.full(n_agents, r0, dtype=float)
        self.excluded = np.zeros(n_agents, dtype=bool)

    def update(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """r_i(t) = lambda*r_i(t-1) + (1-lambda)*(1 - a_i(t))  [Section 5.1,
        the corrected sign convention]. Returns the updated trust vector."""
        assert anomaly_scores.shape == (self.n_agents,)
        self.r = self.lam * self.r + (1 - self.lam) * (1.0 - anomaly_scores)
        self.r = np.clip(self.r, 0.0, 1.0)
        # hysteresis exclusion (Section 5.3)
        newly_excluded = (~self.excluded) & (self.r < self.r_min)
        reentered = self.excluded & (self.r > self.r_reentry)
        self.excluded = (self.excluded | newly_excluded) & ~reentered
        return self.r.copy()

    def coordination_weights(self) -> np.ndarray:
        """r_i, used directly for h~_i = r_i * h_hat_i (Section 5.3)."""
        w = self.r.copy()
        w[self.excluded] = 0.0
        return w

    def strategic_weights(self) -> np.ndarray:
        """u_i = r_i^gamma (Section 5.4); gamma >= 1 allows stronger
        strategic-layer suppression than the coordination layer without
        changing coordination-layer behavior."""
        w = self.r ** self.gamma
        w[self.excluded] = 0.0
        return w
