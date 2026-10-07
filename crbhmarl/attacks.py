"""
attacks.py — Byzantine attack implementations, Section 4.3 / 6.3.
Each attack function takes the true fitness/gradient of a compromised
agent (plus, for adaptive attacks, an estimate of the honest population)
and returns the REPORTED (possibly falsified) value. Honest agents never
call these -- callers should apply them only to the indices in the
Byzantine set B for the current episode.
"""
from __future__ import annotations
import numpy as np


def a1_random_fitness_noise(h_true: float, sigma_b: float, rng: np.random.Generator) -> float:
    """A1: h_hat_i = h_i + N(0, sigma_b^2)."""
    return h_true + rng.normal(0, sigma_b)


def a2_fitness_inflation(h_max: float) -> float:
    """A2: h_hat_i = h_max, regardless of true fitness."""
    return h_max


def a3_sign_flip_gradient(g_true: np.ndarray, c: float = 1.0) -> np.ndarray:
    """A3: g_hat_i = -c * g_i."""
    return -c * g_true


def a4_naive_colluding(h_true: float, g_true: np.ndarray, h_max: float,
                        c: float, sigma_b: float, rng: np.random.Generator
                        ) -> tuple[float, np.ndarray]:
    """A4: combine A2 (fitness) + A3 (gradient) without adapting to the
    detector."""
    return a2_fitness_inflation(h_max), a3_sign_flip_gradient(g_true, c)


def a5_stealthy_fitness(h_true: float, tau: float, rng: np.random.Generator,
                         bias_fraction: float = 0.8) -> float:
    """A5: h_hat_i = h_i + delta_i, |delta_i| < tau. Uses a *persistent*
    biased delta (not zero-mean noise) close to the detection threshold,
    to test whether the temporal-consistency signal (a_i^temp) catches
    sustained small bias rather than only erratic behavior -- this is
    exactly the attack Proposition 2's mu_B > mu_H assumption is not
    guaranteed to survive; see the RQ5 discussion in Section 7.5."""
    delta = bias_fraction * tau  # persistent, not resampled every call by
    # design -- callers should keep delta fixed per-episode per agent to
    # simulate a *consistent* stealthy bias rather than noise.
    return h_true + delta


def a6_adaptive_gradient(g_true: np.ndarray, mu_honest_est: np.ndarray,
                          epsilon_b: float, rng: np.random.Generator,
                          alpha: float = 1.0) -> np.ndarray:
    """A6: g_hat_i = g_i + alpha*v_i, constrained so that
    ||g_hat_i - mu_honest_est|| <= epsilon_b. v_i is chosen (here, via a
    random direction reproducible per-call) to maximize degradation while
    staying inside the epsilon_b ball around the attacker's estimate of
    the honest gradient mean -- i.e., statistically camouflaged rather
    than an obvious outlier."""
    d = g_true.shape[0]
    v = rng.normal(size=d)
    v = v / (np.linalg.norm(v) + 1e-8)
    candidate = g_true + alpha * v
    # project back into the epsilon_b ball around mu_honest_est if violated
    offset = candidate - mu_honest_est
    norm = np.linalg.norm(offset)
    if norm > epsilon_b:
        offset = offset / norm * epsilon_b
    return mu_honest_est + offset


ATTACKS = {
    "random_fitness": "a1",
    "fitness_inflation": "a2",
    "sign_flip": "a3",
    "colluding": "a4",
    "stealthy": "a5",
    "adaptive": "a6",
}
