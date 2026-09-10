# Paper-validation reproduction audit — 2026-09-10

## Outcome and scope

The original validation METHOD was reproduced on the frozen ten-account diagnostic
cohort. The full paper RUN was not reproduced: its results JSON, run log, exact profile
manifest, source revision and configuration were not found. Do not describe these results
as reproducing 51,369 observations or replacing the paper's 42.2/52.3/74.9% figures.
The user has been asked where those original artifacts are stored.

This comparison explains why the prior strict-HF diagnostic appeared much lower. The
legacy combined score includes dust classification and low-effective-LT rules, as well
as adjusted HF. On this cohort its baseline is 8/11 (72.73%), of which seven successes
are dust classifications. The prior diagnostic's 1/11 measures a different criterion.

With snapshots the combined score becomes 7/11 (63.64%), while static-HF detection of
distinct events increases from 2/11 to 4/11. These results do not establish a degradation
of state fidelity: four previously classified dust events become regular events. Some
baseline dust successes had zero reconstructed debt. The enriched data removes that
basis for claiming successful detection. Neither combined score is a validated protocol-
eligibility or prospective prediction metric.

## Matched experiment

Original code: be8a6f7bf66153ecc03b8c1f048fdcafb66a60e1.
Fixed protocol code before this audit: b85953cc9ad36318d75041bfee3f841b6014f49a.
Both were extracted to isolated directories; original outputs and manuscripts untouched.
All variants retain the same ten profile files and SHA256 identities from
`docs/revision-evidence/replay-cohort.json`, eleven held-out observed liquidations,
profile order, profile wallets, profile initial-position setup, cached price history,
static reserve configuration, synthetic pool liquidity, oracle delay 300 seconds,
base margin 1.10, configured volatility discount 0.08, and original overlapping
liquidation-focused split. This is a diagnostic cohort, not the original paper cohort.
The script's 8% discount differs from the paper's stated 10%; without the run configuration,
neither should be assumed to be the actual published experiment setting.

| Protocol / data | Strict distinct events | Static distinct events | Legacy combined | Dust / predicted dust | Failed core actions |
|---|---:|---:|---:|---:|---:|
| Original / core | 1/11 | 2/11 | 8/11 | 7/7 | 106 |
| Original / snapshots | 1/11 | 4/11 | 7/11 | 3/3 | 100 |
| Previously fixed / core | 1/11 | 2/11 | 8/11 | 7/7 | 114 |
| Previously fixed / snapshots | 1/11 | 4/11 | 7/11 | 3/3 | 115 |

All histories, including failures, remain in each comparison. More rejected actions under
stricter feasibility rules do not by themselves establish worse simulation fidelity.
Snapshots are strictly prior-block and strictly earlier timestamp, with unchanged wallet
funding. No current liquidation label is inserted as an anchor. However, this forensic
experiment deliberately retains the legacy scorer's state-selection and label-dependent
policies; it is NOT the corrected pre-event evaluation from the earlier enrichment study.
The legacy scorer may use the state two transactions back, and evaluates enhanced views
with the simulator already at the end of the replay. Anchoring the executor does not fix
those scoring limitations. Do not use these scores as independent historical ground truth.

## Counting defect reproduced and corrected

The legacy validator appends `predicted_hf_before_liquidation` and delayed HF twice per
liquidation; the plotting code counts both entries against the event denominator. For
these eleven events it emits 22 HF entries. The original figure calculation therefore
reports strict 2/11 and static 4/11 for baseline, versus distinct-event counts 1/11 and
2/11. For snapshots it reports static 8/11 rather than 4/11. The combined/dust counters
are separate and are not halved by this correction.

There are also duplicate debt-to-cover entries and extra time-gap entries in the enhanced
helper. The implemented correction records each once. Two focused tests verify cardinality,
distinct values and enhanced-helper behavior. The event-cardinality test fails on the
original revision (4 HF entries for two events); all 18 simulator tests pass after the fix.
Matched corrected runs preserve all six aggregate event counters after removing duplicate
HF entries, all replay states, and transaction failures. This is a reporting/alignment fix,
not a change to protocol behavior. Old result files remain untouched; rerun them or use
an explicitly verified event mapping, never silently halve an arbitrary archived array.

The code defect is established in the available original checkout. Its effect on the
published 42.2% and 52.3% is UNVERIFIED until the original run/source is recovered. Do not
halve those paper figures speculatively.

## Event traces and lineage evidence

`event-comparison.json` retains all eleven event comparisons. Examples:

- Account 0x5de64f9503064344db3202d95ceb73c420dccd57, timestamp 1713043597:
  baseline reference state has $0 debt/$0 collateral and receives dust-predicted credit;
  snapshot replay has approximately $265,358 debt/$805,363 collateral and is not detected
  by the legacy scorer. This illustrates why losing dust credit need not mean worse data.
- Account 0xcffabd8aa79802c96244d2d46245a91b9273dd91, timestamp 1686370900:
  legacy strict HF sentinel 1000 becomes 1.08984 with snapshots; combined prediction changes
  from missed to HF-detected. Sentinel 1000 is the validator's infinity replacement.

A complete read of the available `liquidated_profiles/profiles` directory found 13,998
files, 1,331,784 transactions, 38,986 liquidation observations and 13,990 files containing
liquidations, with zero read errors. This does not match the paper's 51,369 held-out count.
The earlier directory audit also found 13,990 overlapping filenames between the two profile
directories; the original validator concatenates both lists without deduplicating accounts.
We did not assume that every duplicated file is identical or invent a deduplicated paper
cohort. The full non-liquidated-directory content scan was stopped; its totals are not
claimed. `inventory.json` preserves the completed positive-directory manifest and counts.

Searches found no original validation results/reports in the journal checkout or the
IDEA_DeFi_Research mirror. The expected `Aave-Simulator/results/validation` directory is
absent. Local source is still provisional because the specified journal ZIP is unavailable.
No Graph/RPC calls, paid work, R execution or manuscript changes were made.

## Reproduction

Installed tool: `tools/paper_validation_replay.py`. It deliberately preserves legacy scoring
for forensic comparisons and refuses existing output files. Run from the simulator root:

```bash
python tools/paper_validation_replay.py --root . \
  --cohort docs/revision-evidence/replay-cohort.json \
  --output /tmp/aave-paper-method-current.json
```

This uses the current disjoint split. To reproduce the comparison above, extract the
preserved original source and explicitly freeze its historical split:

```bash
mkdir -p /tmp/aave-paper-original
tar -xzf /home/spadef/data/craft-soc/data/validation/paper-reproduction-20260910/original-source.tar.gz -C /tmp/aave-paper-original
python tools/paper_validation_replay.py --root . \
  --cohort docs/revision-evidence/replay-cohort.json \
  --freeze-split --legacy-root /tmp/aave-paper-original \
  --variant snapshots \
  --evidence-dir /home/spadef/data/craft-soc/data/validation/enrichment-20260910 \
  --adapter-path analysis/indexed_history.py \
  --output /tmp/aave-paper-method-snapshots.json
PYTHONPATH=. python -m unittest discover -s tests -v
```

Use `--variant baseline` without evidence/adapter arguments for the paired core run.
For the exact old-code run, point `--root` at the extracted original source and provide its
`data/reserves/price_history.json` as a symlink to the unchanged installed price cache.
`--freeze-split` is unnecessary when the root already contains the original split.
Original source archives, all six reports, source hashes, event traces, test logs and cohort
inventory are stored separately under `validation/paper-reproduction-20260910/`.

## Reviewer/manuscript status and next dependency

| Concern | Evidence/status |
|---|---|
| R1 simulator assumptions/sensitivity | Matched data/logic ablation completed on diagnostic cohort; full paper run still unverified |
| R1 cohort/exclusions | Positive directory audited; original held-out manifest and denominator unresolved |
| R2 assumption/validation reliability | Double-counting fixed; dust/state-selection/label dependence documented; independent validation unresolved |
| Precision/recall/negative cases | Legacy scorer evaluates observed positives only; its label-dependent dust branch cannot be honestly assigned prospective FP/TN counts. Earlier enrichment report retains separately defined confusion matrices |
| Policy/model/cost comparisons | Deferred until simulator validation; no new broad-performance claims |

Required next source: original `validation_results_*.json` and corresponding run log/config
(or an exact profile manifest, code revision and input-cache identities). Recover those,
recompute figure numerators against stable unique event IDs, then rerun the same full cohort
with data/logic variants. Preserve duplicated profile observations separately from unique
accounts/events. Check strict, static, HF-only dynamic, dust and LT-union outputs separately.
Manuscript/reviewer notes should flag counting, definition and lineage checks as pending;
no existing number, figure or cover-letter claim was edited in this phase.
