# CR-BH-MARL — V11 experiment code (IEEE 33-bus, exact AC power flow)

Reproduces every number, table and figure in Section 7 of the V11 manuscript.

- `crbhmarl/reputation.py`, `robust_agg.py`, `attacks.py` — unchanged from the uploaded versions.
- `crbhmarl/ieee33_fast.py` — IEEE 33-bus feeder (case33bw data) with an exact backward/forward-sweep AC
  power flow; `validate()` checks it against pandapower (max |dV| = 1.6e-9 pu).
- `crbhmarl/bench.py` — 9 deployable methods + independent stores + oracle, attacks A1–A6 (A6-PB / A6-WB),
  three-component DSR, recall / FPR / BIR, trust trajectories.
- `run_part.py main <m1,m2,...>` — main benchmark (30 paired seeds) -> `res/`.
- `jobs.py <seconds>` — resumable queue: breakdown refinement, signal ablation, non-IID, topology,
  sensitivity, severity sweeps, scalability, trust trajectories -> `jobs/`.
- `analysis.py` — statistics (paired bootstrap, exact McNemar, Holm, GEE) and Figures 7–13
  -> `results.json`, `figs/`.

Install: `pip install numpy scipy statsmodels pandas matplotlib pandapower` (pandapower only for validation).
Full reproduction: 32,370 episodes, ~25 min on one CPU core.

Run order:
    python run_part.py main non_robust,median,trimmed_mean,krum,multi_krum,reputation_only,anomaly_filtering_only,robust_strategic_only,crbhmarl,independent_stores,oracle
    python jobs.py 100000
    python analysis.py
