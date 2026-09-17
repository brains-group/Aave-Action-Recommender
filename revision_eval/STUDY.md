# Full reviewer study

`build_cohorts.py` samples normal users first without inspecting outcome labels,
then selects shortest observed time-to-liquidation stress users outside that sample.
Both are chronological test samples, not claims of unseen training accounts.
The frame requires an available profile and liquidation-target feature row: the
historical source supplies Borrow/Deposit/Repay/Withdraw origins; its
Liquidated→Liquidated table is empty. One checkpoint per user and no cross-cohort
user overlap. The old 12,000 recommendation run is preserved separately.

The frozen study uses 200 users per cohort, test start 1734263316, dataset end
1755726959, 600-second action latency, seven-day common follow-up end, and $1,000
gross intervention capital. The normal frame includes users who later liquidate.
Zero reconstructed debt is retained, not silently excluded. Source and model
hashes, account IDs, generation errors and execution failures are retained.

## Commands

Run from the project root (data/profiles links must resolve):

```bash
python -m revision_eval.build_cohorts --data data --profiles profiles \
  --output NEW_COHORT_DIR --size 200 --test-start 1734263316 --end 1755726959
python -m revision_eval.study --run NEW_RUN_DIR --cohorts NEW_COHORT_DIR \
  --data data --models cache --coverage COVERAGE_JSON --assets ASSET_MAP_JSON \
  --workers 2 --generate-only
python -m revision_eval.study --run NEW_RUN_DIR --cohorts NEW_COHORT_DIR \
  --data data --models cache --coverage COVERAGE_JSON --assets ASSET_MAP_JSON \
  --workers 2
python -m revision_eval.report NEW_RUN_DIR
python -m revision_eval.trend_grid NEW_RUN_DIR
python -m revision_eval.regimes NEW_RUN_DIR --training-cutoff 1734263316
```

`--history-file` reuses a verified prepared feature-history pickle (must cover all
selected users through their cutoff); a fresh run is required if inputs/code change.
`--prepare-only` stops after preparation and provenance. No collection API calls.
The latest fitted date common to all 24 modeled pairs before the test start is
frozen for all users (available artifacts select 2024-12-01 15:33:51.800000).
Missing models fail explicitly. Training is not performed on inference subsets.

## Recommendation/evaluation semantics

Each cohort/data variant exports evaluator-compatible `*-recommendations.pkl`
dictionaries for control, static_hf, repay_only, deposit_only, automation_style,
and agent. Generation failures are placeholders for every policy, not dropped
case IDs. The runner calls `performSimulations.process_recommendation` for each.
It intercepts simulation calls only to apply identical funding settings, a common
absolute end time and the explicitly selected eligibility/warning policy.
`core` and `indexed` variants retain separate inputs/caches; no silent fallback.
Each case keeps the requested and actually applied recommendation, both arms,
coverage, errors and accounting. The original evaluator's filtered statistics are
retained alongside all-paired and all-eligible counts. Abstentions stay in the
cohort. Seven-day delays within a seven-day window cannot establish long-term
prevention; longer follow-up is needed to reproduce the paper's delay endpoint.

The comparison projects positions/prices after the checkpoint. It does **not**
replay identical future recorded user actions. It therefore does not replace the
paper's original replay claims without that distinction. An HF eligibility flag
is distinct from an observed liquidator execution. Mismatch traces preserve this
interpretation; reconstructed prices/states are not independent on-chain truth.

Agent funds and rule-based policy funds are drawn from the same pre-projection
wallet and gross budget. No automatic conversion or funding top-up is added.
The automation-style comparator is a one-decision repay-first heuristic, not a
replica of recurring collateral-sale systems. Gas scenarios are explicit cost
assumptions, not measured fees or balance-deducted execution constraints.

## Bounded sensitivity design

The chain runs core-data sensitivities on the first 20 frozen users per cohort,
with the same subset used for its reference run. Settings are not selected from
held-out performance: wallet multipliers 2/3 and 4/3 relative to stored inferred
funds; zero discretionary intervention budget; static HF-warning margins 1.05
and 1.10; dynamic warning margin with base 1.10. Funding multipliers affect replay
feasibility and those failures must be reported. Warning outcomes are warning
avoidance, **not** alternative protocol liquidation rules. Dust reporting slices
use explicit $1/$10 checkpoint-debt thresholds without changing eligibility.
No broad Cartesian sweep or new deployment is launched.

`trend_grid.py` checks the 21^3 coefficient grid using the exact polynomial form
of the existing trend score, preserves missing predictions as errors, and records
whether group-wise flips are nonmonotonic. It uses new cohort predictions, so its
score is not a reproduction of the old cohort's 0.9353. Grid points are fixed;
this does not prove continuous-space robustness.

## Predictive comparison

`python -m revision_eval.predictive_benchmark --data data --output NEW_OUTPUT \
 --cutoff 1734263316 --end 1755726959`

Per populated liquidation-target task: deterministic 10,000 training and 2,000
test rows; administrative training censoring at cutoff; preprocessing fitted on
training only; KM, penalized Cox PH, and 200-round depth-3 XGBoost-Cox. This is a
compact new model comparison, not a rerun of the original frozen agent models.
Cox numerical fallback tries penalties 1/10/100 based solely on convergence, never
test scores; all attempts/warnings are saved. Empty source tasks are unavailable,
not fabricated. Numerical zero-duration floor: 1e-6 seconds. Record historical
feature-lineage limitations when interpreting the benchmark.

Metrics: precision/recall, known-horizon-label AUC/AP/calibration/Brier, plus
training-censoring-estimated Uno C, cumulative/dynamic AUC, and IPCW Brier/IBS at
1/3/7 days. Undefined estimates remain unavailable. Calibration of the known-label
subset is not a censoring-adjusted population calibration claim. The existing
69.11% variable-follow-up agent-flag statistic remains separate.

`regimes.py` uses trailing seven-day WETH log-return volatility with cut points
estimated before test start, fixed debt-size bins and past-30-day core-checkpoint
activity. `plot_metrics.py` produces readable standalone diagnostic PDFs.
