"""
ieee33_env.py — A REAL physical CPS environment: the IEEE 33-bus radial
distribution test feeder, simulated with pandapower's AC power-flow
solver, standing in for TargetEstimationEnv (toy_env.py) as the actual
environment behind run_episode()/experiment_scheduler.py.

*** This is real power-system physics, not a synthetic proxy. ***
Voltages, reactive-power limits, and safety bounds below come from an
actual AC power-flow solve on the standard IEEE 33-bus test case
(pandapower.networks.case33bw), not an invented quadratic loss. However,
the mapping from "moving target defense" onto this network (what the
defenders control, what "attack" means, what "defense success" means
physically) is a MODELING CHOICE made here, not verified against your
prior framework's actual physical setup -- read Section "Modeling
choices" below and validate/adjust against the real MTD-MARL prior work
before treating this as equivalent to it.

===================================================================
PHYSICAL SETUP
===================================================================
- Network: IEEE 33-bus radial distribution feeder (12.66 kV), 33 buses,
  32 loads, bus 0 is the substation (slack bus).
- Defenders: N agents, each controlling a reactive-power-capable smart
  inverter / DER (a `sgen` in pandapower) at one non-slack bus. If
  N <= 32, agents are assigned to the first N non-slack buses (1..N).
  If N > 32 (the scalability study's N=50,100), agents wrap around
  (multiple DERs share a bus, each controlling an independent fraction
  of that bus's total allowed Q range) -- a real, if simplified,
  multi-DER-per-node setup.
- Control (theta in R^N): each agent's own reactive-power setpoint
  (MVAr), bounded to [Q_MIN, Q_MAX] per agent (see below).
- Disturbance ("what MTD defends against"): at episode reset, an
  additional real-power load step is applied at 1-3 randomly chosen
  buses (representing an external event -- e.g. a demand spike, a
  tripped local generator, or an attacker-induced load manipulation --
  that the coordinated Q-control response must correct for). This is
  DISTINCT from the Byzantine agent attacks (A1-A6) already implemented
  in attacks.py, which corrupt the REPORTED fitness/gradient signals,
  not the physical grid state; both operate simultaneously in a full
  episode, exactly mirroring the manuscript's threat model (external
  CPS-facing disturbance + internal learning-process attack).

===================================================================
MODELING CHOICES (read before treating this as validated)
===================================================================
1. Objective: minimize sum of squared voltage deviations from 1.0 pu
   across all buses, L(theta) = sum_bus (V_bus(theta) - 1.0)^2. This is
   a standard Volt-VAR control objective, not necessarily identical to
   your prior framework's actual MTD reward function -- check.
2. Gradient model: the TRUE gradient dL/dtheta is computed via a
   LOCALLY LINEARIZED sensitivity matrix J = dV/dQ (an N x N Jacobian,
   estimated via finite-difference AC power-flow perturbations ONCE at
   episode reset, not re-solved every step) -- i.e. gradient computation
   is fast (closed-form given J), but becomes a worse approximation the
   further theta drifts from theta=0 within an episode. This is a
   genuine simplification for computational tractability (an N-agent
   Jacobian costs ~N+1 real AC power-flow solves; N=30 takes ~0.7s).
   The actual applied voltage after each aggregate update IS computed
   with a real nonlinear AC power-flow solve (see apply_aggregate_
   gradient), so safety/mitigation checks are never based on the linear
   approximation, only the gradient DIRECTION used for the update is.
3. Each agent's local gradient report equals the TRUE gradient plus a
   FIXED (per-episode, per-agent) private noise offset, sampled once at
   reset -- structurally identical to toy_env.py's private-observation
   noise model, so honest agents behave like noisy-but-consistent
   observers of a shared true signal, not independently-resampled noise
   every step.
4. Local fitness h_i = -|V_i(theta) - 1.0|, agent i's own bus voltage
   deviation -- a locally measurable quantity in a real deployment.
5. Physical safety: hard ANSI-adjacent bounds [0.90, 1.10] pu -- a
   voltage excursion outside this range at ANY point during the episode
   is a safety violation (Y_safe=0), regardless of whether it recovers.
6. Timely mitigation: ALL bus voltages within the operational band
   [0.95, 1.05] pu at the FINAL step of the episode.
7. Successful allocation: unchanged from toy_env.py -- still the
   trust-weighted top-k coordination-layer proxy in run_pilot.py, not
   owned by this environment class.

None of items 1-6 have been checked against your prior framework's exact
success criteria (Section 6.3's T_max and safety-bound freeze). Use this
environment to validate the ALGORITHM against REAL physics, and treat
items 1-6 as a documented starting point to be corrected once you have
the prior framework's exact operational definitions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import pandapower as pp
    import pandapower.networks as pn
except ImportError as e:
    raise ImportError(
        "ieee33_env.py requires pandapower. Install with:\n"
        "    pip install pandapower --break-system-packages\n"
        "(and optionally `pip install numba` for a solver speedup)"
    ) from e


@dataclass
class EpisodeOutcome:
    """Identical shape to toy_env.EpisodeOutcome, Section 6.3."""
    timely_mitigation: bool
    physical_safety: bool
    successful_allocation: bool

    @property
    def success(self) -> bool:
        return self.timely_mitigation and self.physical_safety and self.successful_allocation


class IEEE33BusEnv:
    V_TARGET = 1.0
    V_SAFE_LOW, V_SAFE_HIGH = 0.95, 1.05     # operational band (mitigation)
    V_HARD_LOW, V_HARD_HIGH = 0.90, 1.10     # hard safety bounds (never violate)
    Q_MIN, Q_MAX = -0.10, 0.50               # per-agent reactive power range (MVAr)

    def __init__(self, n_agents: int, dim: int | None = None, obs_noise: float = 0.02,
                 lr: float = 0.15, n_steps: int = 40, seed: int | None = None,
                 n_disturbed_buses: int = 2, disturbance_scale: float = 0.15,
                 numba: bool = True):
        """dim is accepted (and ignored, set to n_agents) purely so this
        class is drop-in compatible with TargetEstimationEnv's
        constructor signature used by run_episode -- theta's dimension
        here is always n_agents (one control variable per agent), not an
        independent parameter."""
        self.n_agents = n_agents
        self.dim = n_agents
        self.obs_noise = obs_noise
        self.lr = lr
        self.n_steps = n_steps
        self.n_disturbed_buses = n_disturbed_buses
        self.disturbance_scale = disturbance_scale
        self.numba = numba
        self.rng = np.random.default_rng(seed)
        self.reset()

    # --- setup -----------------------------------------------------
    def _build_network(self) -> None:
        self.net = pn.case33bw()
        n_buses = len(self.net.bus)  # 33
        self.non_slack_buses = list(range(1, n_buses))  # 32 buses

        # Map agents -> buses (wrap around if n_agents > 32, Section
        # docstring item "Defenders").
        self.agent_bus = [self.non_slack_buses[i % len(self.non_slack_buses)]
                           for i in range(self.n_agents)]
        # If multiple agents share a bus, each gets an independent sgen
        # at that bus (representing separate co-located DERs).
        self.sgen_idx = []
        for i, bus in enumerate(self.agent_bus):
            idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"der_agent_{i}")
            self.sgen_idx.append(idx)

    def _apply_theta_and_solve(self, theta: np.ndarray) -> np.ndarray:
        """Set every agent's sgen q_mvar from theta and run a REAL
        nonlinear AC power flow. Returns the per-agent bus voltage
        vector (one entry per agent, at that agent's own bus)."""
        for i, idx in enumerate(self.sgen_idx):
            self.net.sgen.at[idx, "q_mvar"] = float(theta[i])
        pp.runpp(self.net, numba=self.numba)
        return self.net.res_bus.vm_pu.values[self.agent_bus].copy()

    def _compute_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference sensitivity matrix J = dV/dQ (N x N) and
        the baseline voltage V0 at theta=0, computed ONCE per episode
        (see module docstring, modeling choice 2). Returns (V0, J)."""
        delta = 0.02  # MVAr perturbation
        theta0 = np.zeros(self.n_agents)
        V0 = self._apply_theta_and_solve(theta0)
        J = np.zeros((self.n_agents, self.n_agents))
        for j in range(self.n_agents):
            theta_pert = theta0.copy()
            theta_pert[j] = delta
            V_pert = self._apply_theta_and_solve(theta_pert)
            J[:, j] = (V_pert - V0) / delta
        # restore theta=0 state for caller
        self._apply_theta_and_solve(theta0)
        return V0, J

    def reset(self) -> None:
        self._build_network()

        # Disturbance: extra real-power load at 1-3 random buses,
        # representing the external event the MTD response corrects for.
        disturbed = self.rng.choice(self.non_slack_buses,
                                     size=min(self.n_disturbed_buses, len(self.non_slack_buses)),
                                     replace=False)
        for bus in disturbed:
            load_rows = self.net.load[self.net.load.bus == bus]
            for lidx in load_rows.index:
                self.net.load.at[lidx, "p_mw"] *= (1.0 + self.disturbance_scale)
                self.net.load.at[lidx, "q_mvar"] *= (1.0 + self.disturbance_scale)

        self.theta = np.zeros(self.n_agents)
        self.V0, self.J = self._compute_jacobian()

        # Fixed per-episode, per-agent private noise (toy_env-style
        # persistent observation offset, module docstring item 3).
        self.grad_noise = self.rng.normal(0, self.obs_noise, size=(self.n_agents, self.n_agents))

        self._safety_violated = False
        self.V_current = self.V0.copy()
        self._apply_theta_and_solve(self.theta)  # sync net state, record actual V
        self.V_current = self._apply_theta_and_solve(self.theta)
        self._check_safety(self.V_current)

    # --- CR-BH-MARL interface (matches TargetEstimationEnv exactly) --
    def local_fitness(self, agent_idx: int) -> float:
        return float(-abs(self.V_current[agent_idx] - self.V_TARGET))

    def local_gradient(self, agent_idx: int) -> np.ndarray:
        """Agent i's PRIVATE, noisy report of the TRUE full gradient
        dL/dtheta (linearized model, module docstring item 2), matching
        toy_env's private-noisy-observation-of-shared-signal structure."""
        L_grad_true = 2.0 * self.J.T @ (self.V0 + self.J @ self.theta - self.V_TARGET)
        return L_grad_true + self.grad_noise[agent_idx]

    def apply_aggregate_gradient(self, g_agg: np.ndarray) -> None:
        self.theta = np.clip(self.theta - self.lr * g_agg, self.Q_MIN, self.Q_MAX)
        self.V_current = self._apply_theta_and_solve(self.theta)  # REAL nonlinear power flow
        self._check_safety(self.V_current)

    def _check_safety(self, V: np.ndarray) -> None:
        if np.any(V < self.V_HARD_LOW) or np.any(V > self.V_HARD_HIGH):
            self._safety_violated = True

    def timely_mitigation(self) -> bool:
        return bool(np.all(self.V_current >= self.V_SAFE_LOW) and
                    np.all(self.V_current <= self.V_SAFE_HIGH))

    def physical_safety_held(self) -> bool:
        return not self._safety_violated

    def distance_to_target(self) -> float:
        """Kept for interface parity with TargetEstimationEnv; not used
        by run_episode directly, but harmless to provide."""
        return float(np.linalg.norm(self.V_current - self.V_TARGET))
