# Recommendation review and contract fixes — 2026-09-14

The recommendation refresh completed on 2026-09-12 11:15:49 UTC. Its outputs remain untouched in `/home/spadef/data/craft-soc/data/evaluation/refreshed-recommendations-20260911`. This is the baseline using simulator `2daaa1d80e233af5d7ea6b999c63d3cab2e462f6`, not a result using the fixes below. Models were held fixed; this was recommendation regeneration, not full model retraining. There were 12,000 generation slots, 34 errors (27 unknown market mappings, seven without a strictly prior snapshot), and 348 abstentions.

## Review of completed recommendations

| HF-only results | Old recommendations | Refreshed recommendations |
|---|---:|---:|
| Core eligible profiles | 4,748 | 4,991 |
| Core rescues / baseline insolvencies | 1,300 / 1,507 (86.26%) | 1,339 / 1,587 (84.37%) |
| Indexed eligible profiles | 5,018 | 5,324 |
| Indexed rescues / baseline insolvencies | 2,106 / 2,490 (84.58%) | 2,188 / 2,630 (83.19%) |

The changed denominators prevent a direct interpretation of those percentage changes. Matching the intersection of the two HF-only subsets by recommendation index and checking account identity gives:

| Matched subset | Profiles | Baseline insolvencies, both runs | Old rescues | Refreshed rescues |
|---|---:|---:|---:|---:|
| Core | 4,703 | 1,496 | 1,293 | 1,297 |
| Indexed | 4,997 | 2,473 | 2,092 | 2,103 |

Thus refreshed recommendations improve rescues slightly on the common filtered cases. Both matched subsets have zero worsening. This intersection is still outcome-filtered and is not an unbiased deployment sample. The refreshed *script-filtered* indexed results contain 181 worsenings, versus zero in its HF-only subset. Do not generalize zero worsening beyond its stated filter. These are paired simulator outcomes, not accuracy against chain truth, and are not directly comparable to the paper's 1,470/1,693 result. Full matched indices and counts are in `matched-review.json`; `compare.py` reproduces this review without modifying results.

## Fix status

| Finding | Implemented / evidence | Remaining limitation |
|---|---|---|
| C01 | HF uses latest price at or before query; no next-price or current-price fallback when history lacks coverage. Missing coverage raises. | Transaction-derived indexed prices are not independent oracle observations. |
| C02 | HF projects reserve copies and local prices, including totals; repeated queries leave live reserves/prices unchanged, including on failure. | Older-than-state queries are rejected; reverse index reconstruction is unavailable. |
| C03 | Recommendation strategies default to positive debt and HF < 1; dust, low LT and margins cannot authorize liquidation. Legacy warning checker explicitly requires `policy='legacy_warning'`. Zero-debt dust and token/USD ratio defects fixed. | Historical `validate_simulator` margin analysis remains a legacy warning diagnostic. Strict HF alone does not verify other configuration gates. |
| C04 | Explicit dated `contract_config` implements audited V3.0–V3.2 50% close factor above HF 0.95 and 100% otherwise. | Production activation timeline missing. Unknown rules and 3.3+ are rejected, not assigned a guessed version. |
| C05 | Explicit collateral bonus and protocol fee; fee applies only to bonus and remains aToken collateral in treasury. No implicit liquidator faucet. | Only underlying payout supported; `receiveAToken` is rejected. eMode-specific parameters require supplied configuration. |
| C06 | Liquidation requires source, version and [start,end) coverage; HF outputs identify configured thresholds and unverified deployment coverage. | No complete dated Polygon reserve/risk-parameter series. Synthetic reserve setup remains a scenario assumption. |
| C07 | No claim of resolution. | Collateral-enabled flags, eMode, isolation, caps, pause/freeze and silo restrictions need both state representation and missing historical settings/events. Full contract feasibility is not established. |
| C08 | Variable debt accrues using V3's third-order compounded approximation; supply remains linear. | Floating-point implementation is not exact RAY arithmetic; historical rate series, stable-rate debt and complete pool state remain missing. |
| C09 | Compatibility-named binary strategy now samples chronologically, including price timestamps; no monotonicity assumption or negative lead time. A positive crossing cannot be outvoted by coarse samplers. Missing coverage cannot become a safe label. | Earliest sampled crossing has declared resolution; it is not exact continuous onset. More checks can increase runtime. |
| C10 | Parent checkpoint adapter rejects affected ambiguous underlying symbols instead of silently aggregating distinct USDC assets. | Address-keyed core profiles/models and verified unknown-market mappings still needed; some formerly accepted cases now fail explicitly. |
| C11 | Snapshot staleness remains explicit in provenance and parent source description. | No verified as-of global index series or stable-rate reconstruction; carried-forward balances remain approximate. No blind second multiplication by snapshot index. |
| C12 | Unsupported actions fail explicitly; outputs list actual supplementary usage. Historical transfer tool applies transfers at their own times. | Active recommendations still use snapshots, not raw transfers/flashloans. Atomic flashloan reconstruction and overlapping receipt reconciliation unavailable; no silent append/double count. |
| C13 | Both replay entry points share an atomic per-action dispatcher. Failure restores financial state; errors are retained. Complete block/log metadata controls ordering; otherwise timestamp-only ordering is explicit. | Whole-transaction bundles and beneficiary/payer distinction require richer profiles/receipts. Per-action rollback is not transaction-bundle atomicity. Direct protocol method calls are not a transaction wrapper. |
| C14 | Price basis, parameter coverage, supplementary usage and reconstruction failures are explicit. | Independent historical oracle/index observations remain missing. Source agreement is not independent validation. |

These changes do **not** close all 14 audit findings. In particular, C07 remains implementation work as well as a historical-data gap. No contract-fidelity or reviewer-resolution claim should be made until those gaps are addressed. Existing collateral-only representation cannot faithfully model a disabled collateral balance. Do not run a new full recommendation campaign treating the remaining approximate state as verified chain state.

## Verification and outcome

31 offline unit tests pass, including new counterexamples for as-of price selection, query purity, missing coverage, zero-debt eligibility, failed-action rollback, compounded debt, version gating, close factor and bonus fee, and non-monotone/negative-time detection. Existing consumer checkpoint tests pass. Python compilation succeeds. No R code, live Graph/RPC requests, billing changes or manuscript edits were made.

The bounded same-input transfer replay has 6,151 transaction checkpoints, 247 observed liquidation positives and 5,904 negatives. Strict counts before and after are TP=41, FP=46, TN=5,858, FN=206: precision 47.13%, recall 16.60%, FPR 0.78%. It does **not** show improved detection. It retains incomplete histories and uses synthetic reserve settings, legacy symbol aggregation and transaction-derived prices; it is a diagnostic, not the performSimulations prevention evaluation. Static/dynamic warning metrics remain separate. See `historical-summary.json` for failure/zero-debt counts and input hashes. Full replay traces remain privately in the session work directory rather than overwriting baseline outputs.

## Commands and next steps

From `Aave-Simulator`:

```bash
MPLCONFIGDIR=/tmp/aave-contract-mpl python -m unittest discover -s tests -p 'test*.py' -q
```

To reproduce matched results from the parent:

```bash
python docs/contract-fixes-20260914/compare.py
```

The script writes a local JSON review beside itself. Historical commands and input hashes are recorded in `replay.py` and `historical-summary.json`; its session-specific paths are intentional evidence references, not portable data defaults. Diagnostics refuse to overwrite existing outputs: select a new output path for another run.

Liquidation scenarios must supply `contract_config` on their transaction (or as a keyword to `sim.liquidate`), with `version` in `3.0`, `3.1`, `3.2`, `source`, `start_timestamp`, `end_timestamp`, and `collateral[asset] = {'bonus': fraction, 'protocol_fee': fraction}`. Tests contain synthetic examples. Do not copy fixture dates/parameters into production configuration. Missing configuration causes an explicit execution failure, so a historical profile containing such liquidations is not a complete reconstruction.

New source hashes invalidate the recommender's simulator cache automatically. Old run directories and action/prediction caches were preserved; a future full recommendation generation still needs a fresh run namespace. Next: obtain/version reserve configuration and index/oracle history, implement collateral flags and mode/feasibility rules, migrate ambiguous assets to address identities, and validate the same cohort before another expensive full recommendation refresh. Track manuscript/reviewer implications privately; no new scientific paper results have been inserted.
