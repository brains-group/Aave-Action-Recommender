# Indexed-data enrichment: 2026-09-10

## Outcome

New position snapshots improve historical balance reconstruction on the fixed diagnostic
cohort, but do not establish reliable liquidation-detection accuracy. Standalone transfer
replay did not improve detection. Do not replace paper results or enable recorded-state
anchors in counterfactual recommendations based on this experiment.

The completed Polygon download contains 1,561,800 transfers, 4,057,345 flash loans and
13,075,112 position snapshots. It used 18,861 of the authorized 30,000 HTTP attempts.
This investigation made no additional Graph/RPC requests and ran no R code.

## Frozen comparison

Baseline simulator: `3da701a708454c6fdf36ea4fc992cc0b20a20000`.
Pipeline: `8143c0bad8ebe16af7b74a8f7f2c0351723d0ca4`.
Cohort: `docs/revision-evidence/replay-cohort.json`, ten accounts, 393 original checkpoints,
11 observed liquidations, 382 other recorded actions. This is the previously selected
stress/diagnostic sample, not representative deployment performance or a new policy study.
All 393 checkpoints matched unique core event identities. Current cached core extraction
has 405 rows for these users; only the original 393 checkpoints are evaluated.

Extracted supplementary evidence: 309 transfers, 31 flash loans, 725 position snapshots.
Snapshots and transfers after a checkpoint are not available to its reconstruction.
All current-block and same-timestamp snapshots are excluded. Related account totals are
not used. Raw balance is divided by token decimals; it is not multiplied by a ray index
again. Sparse balances are carried forward without claiming exact intervening accrual.

All variants use identical cached as-of prices, static reserve parameters, 1e9 synthetic
reserve liquidity, checkpoint identities, and explicit funding presets. Presets are the
existing profile wallet and retrospective minimum funding; both are future-derived capital
assumptions. Neither is a claim about feasible earlier recommendation funding.
The baseline with original profile order reproduces the prior diagnostic exactly.
There are 13 adjacent execution-order inversions in the profiles; source block/log order
is used in all data variants. Correcting order alone leaves aggregate detection counts
unchanged on this cohort. Original profiles and result caches are preserved.

### Immediate observed-event comparison, existing profile wallet

| Data view | TP | FN | Warnings on non-liquidation checkpoints (FP) | TN | Precision | Recall | FPR | Zero-debt liquidations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core replay | 1 | 10 | 7 | 375 | 12.50% | 9.09% | 1.83% | 6 |
| Core + standalone transfers | 0 | 11 | 6 | 376 | 0% | 0% | 1.57% | 6 |
| Prior-block snapshot anchors | 3 | 8 | 61 | 321 | 4.69% | 27.27% | 15.97% | 0 |

An observed liquidation transaction is different from protocol eligibility. The FP column
means disagreement with a transaction execution label, not proof the position was healthy.
No threshold or risk parameter was fitted to labels. Static and dynamic warning policies,
FNR, funding sensitivity and exact counts are in `enrichment-20260910/summary.json`.
With retrospective minimum funding, transfer replay again leaves strict detection unchanged
at 0/11; snapshots detect 3/11. Snapshot pre-state does not depend on inferred wallet funds,
although execution of subsequent actions still can fail.

### Withheld state checks

We compare predicted post-block balances to snapshots from that block, which are withheld
from all anchor inputs for that block. The reference is the same indexed source, not an
independent RPC ground-truth dataset. Only comparable market/side cases are retained:
partial current-block references, unavailable valuation prices, and ambiguous symbol-to-
underlying mappings are excluded. All six variants use the same 358 reference cases and
identical valuation weights. These are asset/side observations, not 358 independent accounts.

| Data view / capital | Mean absolute per-asset USD error | Median absolute USD error |
|---|---:|---:|
| Core / profile wallet | 8,652.59 | 0.898 |
| Transfers / profile wallet | 8,686.17 | 0.906 |
| Snapshots / profile wallet | 2,175.85 | 0.009 |
| Core / retrospective minimum | 644.77 | 0.231 |
| Transfers / retrospective minimum | 604.12 | 0.190 |
| Snapshots / retrospective minimum | 222.14 | <0.001 |

Snapshot anchors reduce mean absolute error about 75% under the existing wallet and 66%
under retrospective minimum funding. This measures consistency with held-out indexed
balances. It does not independently validate historical collateral flags, eMode, oracle
prices, liquidation parameters, close factors, or stable-debt accrual.

For an explicit seven-day warning horizon, exclude the 11 current-liquidation checkpoints
and label the remaining 382 by a subsequent observed liquidation within seven days.
There are 19 positive and 363 negative checkpoints. Snapshot warnings give TP=3, FN=16,
FP=58, TN=305; core replay gives TP=0, FN=19, FP=7, TN=356. Those three warnings precede
one liquidation, with earliest warning lead 183,358 seconds (about 2.12 days). This small,
selected sample does not support a broad predictive-performance claim.

## Implemented safeguards and integration

`analysis/indexed_history.py` provides historical-only prior-block position anchors and
standalone collateral-transfer replay. It rejects snapshot anchoring in counterfactual
mode. Transfers alter supplied positions, not external underlying wallets or pool liquidity;
outgoing transfers cannot silently create negative collateral. Repeated/conflicting source
IDs fail rather than being silently overwritten. Multiple transfers sharing a transaction,
asset and parties are withheld when ambiguous; core-overlapping transactions are withheld.
In this cohort, 55 transfer records overlap core transactions; 245 eligible account-transfer
actions fall before the final checkpoints, of which 237 apply and eight fail under the
original profile funding. This deliberately omits complex overlapping/atomic transactions.

The adapter requires distinct underlying assets to have distinct simulator keys by default.
The legacy diagnostic explicitly permits symbol aggregation to match the existing engine:
bridged and native USDC share a symbol/price history. The state comparison excludes these
ambiguous assets. Production use must resolve asset-address-specific prices and balances.

`profile_gen/user_profile_generator.py` now orders complete metadata by block/log and records
that ordering in new profiles. Missing metadata falls back to stable timestamp order with
an explicit unverified-order marker. Existing profiles are not regenerated silently.

Flash-loan rows are retained for transaction-context diagnostics, not added as persistent
debt or free wallet funding. The downloaded schema lacks enough call-flow/rate-mode detail
to reconstruct all atomic funding and settlement. Historical snapshots must never overwrite
intervention states after a recommendation changes the account's actions.

## Provenance and limitations

Serving deployment: `QmZvndp7kSUaMZo3W21bLyggU8wpcYG5LXBbGvu21t4cvD`.
Coverage manifest:
`/home/spadef/data/craft-soc/data/supplementary/aave_v3/polygon/5a19507ddc6a0f4b/coverage.json`.
The indexed interval ends exclusively at 1788975588. Jobs recovered to different snapshot
blocks during downloading; completed indexed intervals do not certify complete chain history.
The five missing historical borrows and one missing liquidation from the prior audit remain
unresolved and were not inserted into this source.

Source review pinned Messari's public repository to
`2711ac91ef119f321f65b339e10a57f9aa74f9d8`. Its
[transfer handling](https://github.com/messari/subgraphs/blob/2711ac91ef119f321f65b339e10a57f9aa74f9d8/subgraphs/aave-forks/protocols/aave-v3/src/mapping.ts)
accounts for token-event amount conventions, and its
[shared handlers](https://github.com/messari/subgraphs/blob/2711ac91ef119f321f65b339e10a57f9aa74f9d8/subgraphs/aave-forks/src/mapping.ts)
and [position snapshots](https://github.com/messari/subgraphs/blob/2711ac91ef119f321f65b339e10a57f9aa74f9d8/subgraphs/aave-forks/src/sdk/position.ts)
use balance observations. This source revision has not been matched to the deployed WASM;
the paired-record observations are retained as evidence, not treated as definitive event-topic
identification. The [archived V3 aToken implementation](https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/tokenization/AToken.sol)
also distinguishes scaled transfer amounts and token balances. Historical implementation
activation blocks remain required for protocol certification.

Remaining priorities: historical collateral flags/eMode and risk parameters; asset-address
identity; reserve-index/stable-debt accrual; complete liquidation legs; atomic flash-loan
funding; and independent on-chain state checks. More representative or held-out accounts
are needed before recommending these settings for deployment or revising paper claims.

## Reproduction

Run from the simulator root. Frozen extracted inputs and reports are installed separately at
`/home/spadef/data/craft-soc/data/validation/enrichment-20260910/`.

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python tools/enriched_replay.py \
  --cohort docs/revision-evidence/replay-cohort.json \
  --prices data/reserves/price_history.json \
  --evidence /home/spadef/data/craft-soc/data/validation/enrichment-20260910/cohort-evidence.json \
  --assets /home/spadef/data/craft-soc/data/validation/enrichment-20260910/asset-map.json \
  --aligned /home/spadef/data/craft-soc/data/validation/enrichment-20260910/aligned-checkpoints.json \
  --variant snapshots --order source --funding profile \
  --revision "$(git rev-parse HEAD)" --output /tmp/aave-enriched-new.json
```

Choose `baseline` or `transfers` for matched alternatives, and `retrospective_minimum` for
the second capital preset. Existing outputs are refused. To re-extract from caches without
network access, use `tools/prepare_indexed_replay.py --help`; to reproduce paired error and
horizon summaries use `tools/summarize_enrichment.py --run-dir <copied-run-directory>`.

16 tests pass (nine prior simulator tests plus seven new enrichment/ordering tests).
The reusable extraction command reproduced all four evidence inputs exactly; a final replay
reproduced every checkpoint state, metric and failure in the snapshot variant. No manuscript,
cover letter or figures were edited. Reviewer simulator/sensitivity concerns remain partially
addressed; these results do not resolve independent validation or policy-comparison requests.
