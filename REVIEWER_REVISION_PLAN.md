# Major-revision work plan

Decision: major revision and re-review, due **18 October 2026**, using **Create a Revision**.
Source: `ACM_DLT_Journal_DeFi_Lending_Recommendations/reviews/dlt-journal.txt` (AE and both
reviewers, including additional questions). No submission or journal contact is authorized
by this plan. Deliver a clean paper, the same paper with visible changes, and a point-by-point
response/cover letter. This plan does not mark any empirical concern resolved prematurely.

## Reviewer-to-work map

| ID / source | Request | Evidence and concrete remaining action | Completion criterion / status |
|---|---|---|---|
| R1-1; R2-4; AE | Simulator assumptions and fidelity | Finish the running 12,000-recommendation paired rerun. Audit HF warnings versus protocol eligibility, dust/close factors against historical Polygon versions, inferred wallets, action feasibility, oracle/parameter history, and initial positions. Use existing event traces to prioritize real divergences. | Matched state/label errors, exclusions and bounded sensitivity results; **partial, rerun active** |
| R1-2; R2-3; AE | Practical policy baselines | Implement no intervention, static HF-buffer, repay-only, deposit-only and a documented automation-style policy behind one evaluator. Compare to the survival agent with identical states, capital, latency, costs and horizon. Do not label an approximation as a reproduction of a commercial service. | Paired results and uncertainty for every policy on the same frozen cases; **planned** |
| R1-3; R2-3; AE | Representative users and market conditions | Preserve the high-risk/shortest-time sample as a stress test. Add account-level sampling from the full eligible population before looking at outcomes, stratified by activity, position size/assets and market regime. Separate the final temporal holdout and unseen-account analysis. | Cohort flow table, inclusion probabilities, distributions and per-regime results; **partial inventory, broader evaluation planned** |
| R1-4; R2-3; AE | Predictive-model metrics/comparisons | Define target and horizon; audit 69.11% against actual code/denominator. Export frozen model predictions, labels and censoring. Add precision/recall, PR-AUC, ROC-AUC, calibration, Brier score, concordance and time-dependent survival metrics where identifiable. Compare a small set of survival baselines. | Metrics with event prevalence, censoring rules, confidence intervals and held-out splits; **planned** |
| R1 additional experimental comments | Intervention costs | Report capital committed/spent, gas, transaction count, latency, failed actions and any conversion/slippage costs; use identical budgets for every policy. Historical gas evidence where available, otherwise explicitly labeled sensitivity assumptions. | Net outcome/cost table and budget curves, including abstentions; **planned** |
| R2-1; AE; both related-work comments | Methodological novelty and positioning | Build a technical comparison matrix of relevant survival/liquidation predictors, automation policies and agent systems. Distinguish established components from the contribution actually supported by ablations. Verify recent sources before updating bibliography. | Specific contribution statements linked to evidence, not integration alone; **planned** |
| R2-2; AE | Justification of return period and trend score | Check implementation, units, calibration, monotonicity, limitations and relation to survival probability/hazard. Reconcile the two-day example with T=Delta_t/P and the illustrative seven-day horizon. Audit sensitivity search over distance groups against exhaustive bounded evaluation. | Correct definitions and reproducible property/sensitivity checks; **planned; concrete code checks identified** |
| R1-5; R2-5; AE | References, figures, balanced claims | Fix and visibly track the accidental Cref at the evaluation heading; audit remaining labels/citations; redesign hard-to-read panels with explicit units/denominators; replace unsupported execution/causality/fidelity claims only after evidence review. | Both PDF builds, readable figures, claim-to-evidence checklist; **reference correction in this change; remainder planned** |
| R2-3/4; AE | Generalizability and limitations | Distinguish Polygon deployment evidence from other Aave markets/protocols; document data coverage and adaptation requirements. Do not promise a new chain study before free source coverage and budget are established. | Specific scope/limitations and supported portability discussion; **planned** |
| Editors | Revision and explanation of changes | Maintain this map and response notes as evidence lands; draft the response by reviewer item, with section/table/change IDs and open limitations. | Clean + marked PDFs from one source, final response letter and reproducible artifact manifest; **planned** |

## Execution order and bounded scope

### 1. Finish and audit the current run before new large experiments

Use `PAIRED_RERUN.md` and `evaluation/recommendations-20260910/`. Its new cache namespace
preserves old results. Confirm complete.json and finalization.json, then inspect errors,
missing snapshot coverage, same-case denominators, and script-filtered versus unfiltered
outcomes. Do not silently equate the 12,000 saved recommendations with the paper's 4,882
retained profiles. Reconcile 8,400 checkpoints, 5,078 unfiltered completed profiles, 4,882
HF-only profiles, and 1,693 baseline insolvencies using stable recommendation/account/time
IDs and explicit exclusions. Investigate zero-debt cases before dropping them.

Separate prevention from delay, report worsening before and after the historical filters,
and align absolute observation windows for the corrected methodological evaluation.
The current script uses future transactions to set horizons/labels but does NOT replay them;
its active implementation has four time-check strategies. Correct the corresponding paper
claims through tracked changes once the audit is finalized. Do not restart the active run
merely to change prose or analysis summaries.

Output: `evaluation_audit` tables/manifest, paired case ledger, representative divergence
traces and a proposed fixed evaluation specification. Historical/current comparisons remain
separate from claims of verified on-chain ground truth.

### 2. Freeze a defensible evaluation interface and simulator sensitivity set

Core input: account/checkpoint ID, strictly historical position evidence, prices and reserve
parameter version, wallet/capital budget, action latency, cost model and absolute end time.
Policy output: feasible action or abstention, asset/amount, execution time and expected cost.
Result: action success/failure, eligibility timeline, simulated liquidation outcome/time,
ending balances, costs, and explicit reason for missing coverage. Current/future labels
must not force eligibility predictions. Never reset intervention state with later snapshots.

Reuse a control run across policies only when the complete input fingerprint matches.
Treat infeasible recommendations as abstentions/failures within the common denominator,
not automatically as excluded profiles. Keep the old paper-filtered table only as a clearly
identified reproduction view. Report event-level and account-level results separately.

A bounded sensitivity grid, not a Cartesian sweep:
- verified eligibility rules versus the existing warning policy; static versus dynamic
  warnings evaluated as warnings, not alternative protocol liquidation rules;
- observed/explicit funding where available, no-extra-funding scenario, and a declared
  top-up budget; retain retrospective 1.5-style funding only as a labeled sensitivity;
- no collateral discount versus the implemented 8% and manuscript-stated 10%, after
  identifying which settings actually ran; fixed 0/300-second warning-price delay;
- recorded prior positions versus reconstructed positions on the same complete cases;
- historical dust/close-factor behavior only after deployment/version verification, not
  arbitrary HF exceptions copied from V4.

Use one-factor changes around a frozen baseline; predeclare any selected interaction.
Start with a small runtime pilot, estimate cost, then freeze the comparison cohort before
looking at results. Reuse the existing caches and price evidence; no paid collection.

### 3. Data lineage, split and predictive-metric audit (can proceed independently)

Trace raw transaction IDs -> core-event normalization -> task rows -> feature/label pairs ->
training/held-out predictions. Reconcile 8,837,439 historical rows, 21.8M feature-bearing rows,
>10M validation transactions, ~39K raw liquidations and 51,369 held-out observations by source
version and task/profile expansion. Document actual 25 journal tasks (including liquidation
origins and self-transitions), the benchmark-compatible subset, and all 90 feature names.
Supplementary simulator events must not enter survival tasks or historical predictors.

Preserve reference split membership for parity. Check account overlap, equal-timestamp
ordering, feature availability and censoring. No R execution is required for this phase;
use source inspection and existing portable outputs, and explicitly record unresolved parity.

Relevant code: `utils/model_training.py` (preprocessing, baseline hazard, model-by-date),
`utils/data.py` (sampling/history), `actionAgentTraining.py` (prediction and risk decisions),
`analyze_simulation_results.py` (reported accuracy). Export predictions once and reuse them
for metrics rather than retraining for each table. Compare Cox PH and a simple time-to-event
baseline to XGBoost-Cox first; add another complex model only if it answers a remaining
reviewer question. Use validation data to select thresholds, never the final holdout.

For seven-day labels, omit/censor negative checkpoints without sufficient follow-up;
positive events observed within the horizon remain identifiable. Report PR metrics alongside
ROC metrics for imbalance; survival metrics need declared censoring estimation/support.
Use account-clustered uncertainty, not independent-row intervals on expanded task records.

### 4. Risk-metric checks and policy experiments

Inspect `actionAgentTraining.get_expected_time_to_event`, `calculate_trend_slope`, and
`determine_liquidation_risk`; distinguish expected time from inverse-frequency/return-period
interpretations. For T=Delta_t/P with 0<P<=1, T>=Delta_t; P=0 requires an explicit infinite/
undefined convention. Determine the actual horizon before correcting the example.

`sensitivity_analysis.process_sample_row_binary_search` explicitly assumes that absence of
a prediction flip in the most distant coefficient group implies absence in closer groups.
That property is not established. On a deterministic bounded sample, compare all coefficient
groups with the binary-search result, count missed flips and incorrect minimum distances,
and preserve the old results separately. Do not claim an absolute minimum without evidence.

Then run the fixed policy set from the reviewer matrix on the preserved stress cohort and
representative cohort, with common capital/latency/cost budgets. Tune HF buffers only on the
validation split. Report paired prevention/delay/worsening, action feasibility, cost and
performance by market regime and position size; retain all abstentions and failure reasons.

### 5. Manuscript, response and release

Only evidence-supported edits. Add unique change IDs to response notes. Use `\added`,
`\deleted`, `\replaced` (or configured prefixed commands) for text, equations, captions and
citations where safe. For labels/structural changes/assets, keep the operative LaTeX safe,
add a visible changes note, preserve originals and record exact before/after in the change
ledger. Git history is additional evidence, not a replacement for changes markup.

Update abstract/conclusion last, after tables settle. Distinguish observed execution,
protocol eligibility, heuristic warnings, model prediction and conditional intervention
outcomes. Do not claim on-chain execution or broad causal prevention from simulation alone.
Use single-column-readable panels, common axes, uncertainty, explicit denominators and
legends; preserve original figures and map every replacement in the ledger.

## Suggested checkpoints before the recorded deadline

- **11–17 September:** finish current run, reconcile exclusions, freeze evaluation interface
  and simulator assumptions; complete metric/split and return-period/search audits.
- **18–24 September:** implement baseline policies and bounded sensitivity pilot; freeze
  cohorts and evaluation budgets; complete contribution/related-work matrix.
- **25 September–4 October:** run held-out/stress/representative comparisons and cost analysis;
  resolve data/metric discrepancies before selecting manuscript claims.
- **5–11 October:** tracked paper/figure revisions and point-by-point response.
- **12–16 October:** independent consistency review, clean/marked build checks and artifact
  reproduction; retain 17 October as buffer before the **18 October** revision deadline.

These are planning checkpoints, not promises of unmeasured results. If evidence remains
unavailable, narrow the associated claim and state the unresolved limitation. Do not expand
this revision into an uncontrolled multi-protocol study or model redesign.
