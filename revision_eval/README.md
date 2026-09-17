# Journal revision evaluation inputs

This first implementation prepares policy proposals, eligible checkpoint sampling,
fixed-horizon diagnostics and a corrected trend-sensitivity search. It does not
claim that the new policy comparison, representative evaluation, or sensitivity
experiments have been completed. No running evaluation code is modified.

## Policy interface

`python -m revision_eval policies --input checkpoints.json --output proposals.json --budget-usd 1000`

Input is a JSON list. Each checkpoint has `case_id`, evaluation `timestamp`,
`state_timestamp` (must not be in the future), `total_debt_usd`,
`weighted_collateral_usd` (sum of enabled collateral USD times LT), and `assets`.
Each asset key is an unambiguous identity with `price_usd`, `wallet`, `debt`
(token units), `liquidation_threshold` (fraction), and `collateral_enabled`.
These inputs must come from the pre-intervention state, with dated prices and
parameters, not the projected final state or future inferred top-ups.

Policies: no intervention, static HF buffer (cheapest single action meeting the
HF target), repay-only, deposit-only, and repay-first automation-style. Default
trigger 1.10 and target 1.20 are explicit proposed settings, not tuned results.
If a target is unaffordable, propose the largest permitted partial action and
retain `target_reached=false`. Repay uses same-asset funds only; deposits only
use already enabled collateral. No swaps, implicit collateral activation or
commercial-product replication are claimed. Automation-style here means one
checkpoint decision; recurring monitoring/execution is not implemented yet.

Proposals are **not execution-feasibility judgments**. The common simulator must
validate caps, pause flags, accrued debt, transaction ordering, balance changes,
and HF after actual execution. `execution_status=not_executed` prevents treating
proposal generation as a success. No-intervention/abstention cases must remain
in the evaluation denominator. Apply identical wallet, capital, latency, prices,
follow-up, fees and exclusion assumptions to agent and baselines. Match by case
ID, report all eligible cases plus feasible-action subsets separately, and retain
failures. Capital is gross token value committed, not an economic loss. Gas is
unknown unless an explicitly labeled USD assumption is supplied; no zero-cost
claim follows from a missing estimate. This initial policy interface does not
deduct gas from funding, so cost-constrained replay still needs integration.

## Cohort interface

`python -m revision_eval cohort --input eligible.json --output sample.json --sample-size 200`

Rows require unique `case_id`, `account`, `split`, and `eligible`. Only test and
eligible rows enter a deterministic SHA-256 sample. Supply the entire eligible
frame with a recorded historical cutoff; do not call a sample from the original
stress cohort representative. The Python API supports explicit excluded accounts
for account-disjoint tests. Sampling unit is a checkpoint, not a user; high-activity
users can contribute more checkpoints. Future liquidation labels do not influence
selection. Preserve the existing shortest-time-to-liquidation stress cohort
separately. Feature/eligibility creation, full-frame lineage and regime assignment
are outstanding; this function cannot certify them from caller flags.

## Predictive metrics

`python -m revision_eval metrics --input predictions.json --output metrics.json --horizon 604800`

Each row: unique `case_id`, `duration` and `horizon` in seconds, binary `event`,
and event `probability` at exactly that horizon. An event at the horizon counts
positive; survival through the horizon is negative; censoring before the horizon
is excluded and listed. Reports confusion counts, prevalence, precision/recall,
ROC-AUC, average precision (not trapezoidal PR-AUC), Brier score and ten-bin
calibration. Single-class AUC/AP are undefined. These are **known-label-subset**
diagnostics, not IPCW population estimates. Training-only censoring weights,
time-dependent concordance/Brier, Cox and simple time-to-event baselines remain
pending. Agent risk flags are not calibrated probabilities and cannot be passed
as Cox event probabilities to this adapter.

## Sensitivity correction

`sensitivity_analysis.py` now searches coefficient points in ascending actual
Euclidean distance. It checks every nearer point before claiming a minimum; if
none flips it exhausts the grid. Prediction failures stop the run rather than
being counted as stability. Old binary-search caches are rejected; default output
is `cache/sensitivity_grid_v1_results.pkl`. This preserves the original score's
finite-grid definition, not a continuous/global robustness proof. Recompute the
published score before citing it as a corrected result. No expensive rerun was
started while the simulator evaluation occupies the workers.

Tests: `python -m unittest discover -s tests -p test_revision_eval.py -v`.

## Full study runner (2026-09-16)

The integrated cohort, policy, predictive, regime and sensitivity implementation is now described in [STUDY.md](STUDY.md). Its commands supersede the pending-integration notes above. Historical feature-lineage and original-result reproduction remain distinct from the new bounded study.
