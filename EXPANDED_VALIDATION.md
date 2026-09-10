# Expanded Polygon validation — 2026-09-10

The available project data were sufficient to run this comparison; locating a historical report was not a prerequisite. `data` resolves to `/home/spadef/actionAgent/data` and contains transactions.csv, users.csv and the 25 survival task tables. The previous claim that missing run artifacts blocked comparable testing was too broad.

## Fixed comparison

200 deterministic hash-selected accounts were proposed: 100 from the liquidation profile directory and 100 from the background directory, skipping duplicate accounts, with 2–500 transactions each. Two liquidation-group accounts had ambiguous core-event identities and were excluded before any model results. Final cohort: 98 liquidation-group and 100 background accounts, 6,151 uniquely aligned checkpoints, 247 observed liquidation transactions and 5,904 other actions. This is a bounded stress/background comparison, not a population-random or original-paper cohort. All eight runs cover every checkpoint; no variant-dependent exclusions.

Core events retain the existing profile semantics and window; new cached supplementary data are matched to those same histories. Profile files are hashed, original profiles and paper assets are preserved. Funding presets are identical across data variants: existing inferred profile wallet, or retrospective minimum wallet computed from the same original action history. Both use future-derived funding and are sensitivity assumptions, not feasible prospective funding claims. Prices, static reserve parameters, source block/log order and checkpoint identities are held fixed. No thresholds were tuned.

## Strict HF versus observed transaction labels

| Protocol / funding / data | TP | FN | FP | TN | Precision | Recall | FPR | Zero-debt positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-profile | 39 | 208 | 35 | 5869 | 52.70% | 15.79% | 0.59% | 55 |
| transfers-profile | 41 | 206 | 46 | 5858 | 47.13% | 16.60% | 0.78% | 55 |
| snapshots-profile | 32 | 215 | 178 | 5726 | 15.24% | 12.96% | 3.01% | 0 |
| baseline-retrospective_minimum | 59 | 188 | 25 | 5879 | 70.24% | 23.89% | 0.42% | 6 |
| transfers-retrospective_minimum | 75 | 172 | 60 | 5844 | 55.56% | 30.36% | 1.02% | 6 |
| snapshots-retrospective_minimum | 32 | 215 | 178 | 5726 | 15.24% | 12.96% | 3.01% | 0 |
| original-baseline-profile | 53 | 194 | 50 | 5854 | 51.46% | 21.46% | 0.85% | 51 |
| original-snapshots-profile | 32 | 215 | 178 | 5726 | 15.24% | 12.96% | 3.01% | 0 |

FP here means a warning on a non-liquidation transaction, not independently certified healthy protocol state. Protocol eligibility and observed execution are different targets. The old protocol baseline is be8a6f7; current is cdb0bdc. Original-code runs use the same harness and data with the historical-only adapter, not a different validation policy. Core-only strict detection changes from 53/247 with old logic to 39/247 with fixed logic; snapshot detection is 32/247 under both. Stricter action feasibility can reject recorded actions, and rejection counts are retained in each report. These results do not validate either protocol version against independent state truth.

## Withheld indexed balances

The same 5,085 asset/side cases and baseline valuation weights are used across all six current-code variants. Current-block snapshots are withheld from input; only strictly earlier-block and earlier-timestamp snapshots anchor state. This measures agreement with the same indexed source, not independent RPC ground truth. Ambiguous same-symbol underlying assets and partial references are excluded from balance-error comparisons.

| Funding / data | Mean absolute USD error | Median absolute USD error |
|---|---:|---:|
| baseline-profile | 2444.47 | 10.575 |
| transfers-profile | 2646.62 | 9.744 |
| snapshots-profile | 453.32 | 0.062 |
| baseline-retrospective_minimum | 2278.91 | 8.046 |
| transfers-retrospective_minimum | 2053.05 | 4.605 |
| snapshots-retrospective_minimum | 246.45 | 0.011 |

Snapshots reduce mean balance error approximately 81.5% under profile funding and 89.2% under retrospective minimum funding, but reduce immediate strict detection and increase non-liquidation warnings. Transfers raise detected positives from 39 to 41 with profile funding, and 59 to 75 with retrospective minimum funding; their precision falls as warnings rise. Transfer balance error worsens with profile funding and improves with retrospective minimum. Additional data therefore help specific reconstruction failures, but are not an established overall liquidation-prediction improvement. Do not enable snapshot anchoring in counterfactual recommendation simulation.

## Seven-day checks and traces

For a separate forward-looking label, current-liquidation checkpoints are omitted. Negative cases require seven days of recorded follow-up; otherwise they are censored. Positive cases require a subsequent observed liquidation within seven days. This uses recorded follow-up, not independently certified absence of events.

| Data / funding | TP | FN | FP | TN | Censored negative checkpoints |
|---|---:|---:|---:|---:|---:|
| baseline-profile | 3 | 382 | 26 | 5010 | 483 |
| transfers-profile | 9 | 376 | 30 | 5006 | 483 |
| snapshots-profile | 5 | 380 | 137 | 4899 | 483 |
| baseline-retrospective_minimum | 0 | 385 | 21 | 5015 | 483 |
| transfers-retrospective_minimum | 1 | 384 | 30 | 5006 | 483 |
| snapshots-retrospective_minimum | 5 | 380 | 137 | 4899 | 483 |

`paired-traces.json` records gained/lost detections and added warnings, including original event IDs, block/log ordering, prices and ages, balances, snapshot provenance, prior failed actions and the first balance divergence. Raw run reports retain every checkpoint. The summary additionally contains static/dynamic warning counts, funding sensitivity, subgroup counts, failure counts, transfer exclusions and warning lead-time information. The diagnostic dynamic HF policy is not the paper’s dust/LT combined score, and these results must not be compared directly to 74.9%.

## Reproduction and next work

All work was offline using completed caches; no new API requests or R execution. Evidence is installed at `/home/spadef/data/craft-soc/data/validation/expanded-20260910/`. From the simulator root:

```bash
python tools/enriched_replay.py \
  --cohort /home/spadef/data/craft-soc/data/validation/expanded-20260910/cohort.json \
  --prices data/reserves/price_history.json \
  --evidence /home/spadef/data/craft-soc/data/validation/expanded-20260910/evidence/cohort-evidence.json \
  --assets /home/spadef/data/craft-soc/data/validation/expanded-20260910/evidence/asset-map.json \
  --aligned /home/spadef/data/craft-soc/data/validation/expanded-20260910/evidence/aligned-checkpoints.json \
  --variant snapshots --order source --funding profile \
  --revision cdb0bdc0c21b8fa54e50cb357b438c7629e07a2a --output /tmp/expanded-snapshots.json
```

Use `baseline` or `transfers` for alternatives, and `retrospective_minimum` for the funding sensitivity. Set `PYTHONPATH=.` when invoking the script if needed. Existing output files are refused. `run.py`, `run_old.py`, cohort selection, exclusions, frozen raw extracts, summaries and source hashes are preserved with the evidence. Script paths in the orchestration scripts identify this session; the command above uses durable installed paths.

Historical collateral flags/eMode, asset-address-specific prices and historical reserve parameters remain important unvalidated inputs. Flash loans are context only because the available rows do not reconstruct atomic settlement. These results do not justify tuning margins to match labels. The next simulator investigation should use the saved divergence traces, rather than collecting more arbitrary event types or treating dust credit as accuracy. Reviewer assumption/cohort concerns remain partially addressed. No manuscript numbers, figures or cover-letter claims were changed.
