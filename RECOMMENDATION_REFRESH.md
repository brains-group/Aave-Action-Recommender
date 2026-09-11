# Recommendation regeneration, 2026-09-11

The September 10 evaluation reused the 12,000 February recommendations. This run
regenerates candidate predictions and action sizes using the current simulator and
strictly prior indexed snapshots. The original 12,000 source rows in
`cache/train_set.csv` are reordered to the previous recommendation file's order;
user and checkpoint timestamp must match. No cohort resampling.

Changes: funding comes from `checkpoint_state`, never projected final balances.
Repayment uses the same asset's wallet and debt, with no implicit swap. The $50
minimum is converted to token units. No automatic dust repayment upsizing.
Unfunded actions abstain and both evaluation arms use the identical baseline
profile and horizon. Coverage/model failures remain explicit errors, not successful
abstentions. Existing inferred initial wallets remain a retrospective limitation.

The historical survival models and preprocessing are copied into a new generation
cache to isolate the effects of simulator and funding corrections. Supplementary
snapshots are not model features. Missing frozen models fail explicitly; no model
is accidentally trained on the inference-only history subset. Predictions,
recommendations and paired simulations are rebuilt in new directories. Old caches
and prior outputs remain untouched; moving them is unnecessary with explicit cache
paths. Predictions hash the entire candidate feature row, fixing collisions among
same-time, same-amount actions in different assets or of different types.

The updated core transaction CSV is recorded with its SHA256 and a formatter command
in `inputs/training-data.json`. `AAVE_CORE_TRANSACTIONS` supports date boundaries from
that CSV; `AAVE_SURVIVAL_DATA` selects matching formatted training tables and isolates
model caches. Supplementary inputs cannot select survival tasks. New full-data
survival formatting still needs parity/source validation before replacing the
historical models; this run does not claim retraining on those new tables.

Runner: `python refresh_recommendations.py --help`. Supply `--evaluate` to chain
regeneration, both core/indexed evaluation variants and summary generation.
`--prepare-only` builds bounded-memory histories once; `--limit 2` is a pilot and
must use a separate evaluation folder from a full evaluation. Resume with identical
inputs/code. Each recommendation is atomically saved under `generated/`, preserving
errors and timings. `generation-complete.json` reports errors/abstentions;
`complete.json` means the chained evaluation and summary finished, not that every
case was evaluable. Paper filtering/zero-debt exclusions remain visible in results.

Checks: five funding fixtures cover same-asset repayment, USD units, insufficient
funding, future checkpoints and nonfinite prices. Three integration checks also pass: no-op arms are identical, projected wallet is ignored, and different candidate actions cannot collide in the prediction cache. A functional fitted-model/indexed-simulator regeneration check passed; its deliberately empty feature-history series is only a compatibility test, not a study result. A full-history two-case pilot gates the background full run.
Manuscript assets and numbers are unchanged.


Run directory: `/home/spadef/data/craft-soc/data/evaluation/refreshed-recommendations-20260911`.
`chain-status.json` records preparation → pilot → full → complete/failed. Four
regeneration workers and eight evaluation workers. Fresh generation/evaluation
share only this run's code- and input-keyed simulation cache, avoiding duplicate
baseline computation. None of the old simulation caches is selected.
Exact launch arguments are saved in `prepare-receipt.json`; remove `--prepare-only`
and append `--evaluate` to resume the chain manually. Keep the same code/inputs.

## Historical training input check

An offline join by unique core event ID through 2024-12-01 15:33:51.8 UTC (the
last fitted-model date selected by these checkpoints) found 7,030,448 common rows,
zero missing old rows, and 1,815 additional refreshed-source rows. Type, timestamp,
account/liquidatee, pool, reserve, amount and USD amount matched on all common rows
(numerical tolerance rtol=atol=1e-10). This establishes raw-field agreement only,
not whole-pipeline feature/label parity. Additional rows can change engineered
market/account features; the fixed-model comparison must not be called a fully
retrained updated-data study. Full evidence is saved with the run's training inputs.
