"""
toy_env.py — A SYNTHETIC, TOY multi-agent estimation task, standing in for
the real IEEE-33-bus hierarchical MARL-MTD environment (Section 6.2) used
in the actual paper.

*** THIS IS NOT THE REAL EXPERIMENT. *** It exists only so that
run_pilot.py is genuinely executable end-to-end today, to validate that
the CR-BH-MARL algorithmic pipeline (reputation.py, robust_agg.py,
attacks.py) is wired together correctly BEFORE you invest compute in the
real environment. Any numbers this produces are illustrative of the
*pipeline*, not of the paper's actual claims -- do not put them in the
manuscript.

TODO (replace before real experiments): swap TargetEstimationEnv below
for your actual OpenAI-Gym-compatible IEEE 33-bus CPS environment from
the prior Entropy paper (Section 6.2), which provides: local fitness per
defender agent, local gradient contributions from the strategic-layer
actor/critic, and a real defense-success signal per episode (not simple
L2 distance to a synthetic target).

The toy task: N agents each privately observe a shared unknown target
vector theta* with i.i.d. Gaussian noise. The strategic layer maintains
a running estimate theta_t, updated via (robustly aggregated) per-agent
gradients of the local squared-error loss. "Defense success" for an
episode is defined as the final estimate being within a distance
threshold of theta*, loosely analogous to DSR (Section 6.13).
"""
from __future__ import annotations
import numpy as np


class TargetEstimationEnv:
    def __init__(self, n_agents: int, dim: int = 8, obs_noise: float = 0.3,
                 success_threshold: float = 0.5, lr: float = 0.2,
                 n_steps: int = 40, seed: int | None = None):
        self.n_agents = n_agents
        self.dim = dim
        self.obs_noise = obs_noise
        self.success_threshold = success_threshold
        self.lr = lr
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.theta_star = self.rng.normal(size=self.dim)
        self.theta = np.zeros(self.dim)
        # per-agent fixed private observation (noisy view of theta_star)
        self.obs = self.theta_star[None, :] + self.rng.normal(
            0, self.obs_noise, size=(self.n_agents, self.dim)
        )

    def local_gradient(self, agent_idx: int) -> np.ndarray:
        """g_i = grad_theta ||theta - o_i||^2 = 2*(theta - o_i)."""
        return 2.0 * (self.theta - self.obs[agent_idx])

    def local_fitness(self, agent_idx: int) -> float:
        """h_i: higher (less negative) = better local estimate quality."""
        return float(-np.linalg.norm(self.theta - self.obs[agent_idx]))

    def apply_aggregate_gradient(self, g_agg: np.ndarray) -> None:
        self.theta = self.theta - self.lr * g_agg

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.theta - self.theta_star))

    def is_success(self) -> bool:
        return self.distance_to_target() < self.success_threshold
