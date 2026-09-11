# Polygon Aave V3 contract and supplementary-data audit — 2026-09-11

**Conclusion: the current simulator is not a faithful implementation of historical Polygon Aave V3.** Several concrete implementation defects coexist with intentionally heuristic warning rules. Position snapshots are connected to the active recommendation run; downloaded transfers and flash loans are not replayed by that run. This audit documents findings only. It does not change simulator, recommendation, training, collection or evaluation code, cached inputs, configuration, or the manuscript.

The regeneration pilot passed and the full recommendation refresh was active when inspected. Preserve its outputs as the current-code baseline. Do not represent its liquidation labels as verified contract execution outcomes, or interpret this audit as a measured before/after performance result.

## Scope and reproducibility

- Simulator revision: `2daaa1d80e233af5d7ea6b999c63d3cab2e462f6`.
- Recommender revision at audit: `cb81d335ae92ebb2347f6a98575b603d59351762` (documentation commits may subsequently advance the parent).
- Active run: `/home/spadef/data/craft-soc/data/evaluation/refreshed-recommendations-20260911`.
- Supplementary deployment: `QmZvndp7kSUaMZo3W21bLyggU8wpcYG5LXBbGvu21t4cvD`, Polygon; manifest snapshot block **93,513,707**. Completed indexed interval is not proof of complete chain history. The final entity jobs record different source heads (transfers 93,515,384; flash loans 93,522,147; position snapshots 93,559,011), so the shared end time is not a globally atomic source snapshot.
- Eight small diagnostic probes ran in an isolated process, with Python bytecode writes disabled. They are counterexamples/compatibility checks, not an EVM fork or a historical-cohort benchmark. See `probe-results.json` and `probes.py`.
- `simulator-source-manifest.json` hashes the audited Python sources. Source download manifests retain public URLs and SHA256 values. The downloaded Solidity files and GitHub source trees remain in `/home/spadef/.codex/sessions/2026/09/08/contract-audit-20260911/sources/`; they are not added to the simulator repository.
- No Graph API or RPC requests, new collection, billing, heavy simulation rerun, or R execution was performed for this audit. Only public web/GitHub source retrieval and offline inspection/probes.

Reproduce the probes from this project's root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/aave-audit-mpl python docs/contract-audit-20260911/probes.py
```

## Contracts and historical version evidence

The official [Polygon address book](https://github.com/aave-dao/aave-address-book/blob/main/src/AaveV3Polygon.sol) identifies:

| Component | Polygon address |
|---|---|
| Pool proxy | `0x794a61358D6845594F94dc1DB02A252b5b4814aD` |
| PoolAddressesProvider | `0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb` |
| PoolConfigurator proxy | `0x8145eddDf43f50276641b55bd3AD95944510021E` |
| Oracle | `0xb023e699F5a33916Ea823A16485e259257cA8Bd1` |

These proxy addresses do **not** identify one immutable implementation over the whole dataset. The address book retrieved during the audit lists Pool implementation `0x6030dB989D47cD74FC17bB6F4FcD3A8B29FEe57e`; that is a current reference, not a historical specification.

| Source set | What was actually verified | Limit |
|---|---|---|
| [BGD 3.0.2 upgrade](https://github.com/bgd-labs/proposal-3.0.2-upgrade) | Polygon-specific downloaded pre-upgrade Pool logic and pre/post upgrade reports. The pre-report names implementation `0xDF9e4ABdbd94107932265319479643D3B05809dc`; post-report names `0xb77fc84a549ecc0b410d6fa15159C2df207545a3`. | The post-upgrade report is an upgrade-test artifact, not independent proof of the production execution block. |
| [BGD 3.1 upgrade](https://github.com/bgd-labs/protocol-v3.1-upgrade) | Polygon-specific archived Pool logic, deployment script and Polygon fork test at block **59,399,054**; [Origin v3.1.0](https://github.com/aave-dao/aave-v3-origin/tree/v3.1.0) logic also inspected. | The test's fork block is not the activation block. |
| [BGD 3.2 upgrade](https://github.com/bgd-labs/protocol-v3.2-upgrade) | Polygon deployment selection and fork test at block **62,249,338**; [Origin v3.2.0](https://github.com/aave-dao/aave-v3-origin/tree/v3.2.0) logic inspected. Stable-debt offboarding and eMode migration are explicit in its upgrade specification. | Again, fork test is not activation evidence. |
| [Origin v3.3.0 liquidation logic](https://raw.githubusercontent.com/aave-dao/aave-v3-origin/v3.3.0/src/contracts/protocol/libraries/logic/LiquidationLogic.sol) | Later rules introduce selected-reserve size checks at $2,000 and minimum residual rules at $1,000 in the default USD base units; 50% sizing can depend on total account debt. | Cannot apply these retrospectively to 2023–24, or assert Polygon activation without deployment history. |

Immutable Git tree identifiers and fetched URLs are retained in the manifests. **Still missing for a full deployment-level sign-off:** the Polygon proxy upgrade/activation log, implementation bytecode linkage at each boundary, historical reserve/eMode/oracle configuration changes, and coverage of later implementations through every evaluation horizon. The existing code has no implementation-version dispatch or block-version manifest. A contract release tag or current documentation alone cannot close that gap.

## Confirmed defects and contract mismatches

Priority P0 means fix before treating outputs as historically faithful; P1 means an important reconstruction/coverage gap. The listed remedies are proposed work after the active run, not applied patches.

| ID | Priority | Finding and evidence | Consequence / later verification |
|---|---|---|---|
| C01 | P0 | **Future prices in historical HF.** `simulator/protocol.py:1165` uses `bisect_left` and the returned element rather than the last observation at/before the target. A probe with prices at 100/200/300 returned the price from **200 when queried at 150**. `apply_price_updates_at_timestamp` uses a different, as-of algorithm. | Prediction views and transaction replay can disagree and use future information. Test before-first, exact timestamp, between observations, after-last and missing-price cases; carry source/block/age and fail explicitly on unavailable history. |
| C02 | P0 | **HF queries mutate reserve totals.** `get_health_factor_at_timestamp` calls `Reserve.update_state`, then restores indices/timestamps but not `total_liquidity` or `total_variable_debt` (`protocol.py:1142–1236`). Probe: supply **10000 → 10000.8219178**, debt **1000 → 1000.8219178**, while the borrow index returns to 1. | Repeated strategy checks can change subsequent rates/state solely by querying HF. Actual [Pool account-data lookup is a view](https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/pool/Pool.sol). Test query idempotence and strategy-order invariance with the complete reserve state. |
| C03 | P0 | **Warning rules become liquidation outcome labels.** `tools/liquidation_detection_strategies.py:175–235` returns a liquidation flag for dust, adjusted HF below a dynamic margin, or a low effective LT. A zero-debt account is classified as dust; a separate probe reports a low-LT warning at **HF 6.0**. | Historical V3 liquidation validation requires HF < 1, enabled collateral and debt in the selected asset. These are risk-warning policies, not contract eligibility. Separate strict eligibility, warnings and observed liquidator execution. The core `liquidate` method does reject HF ≥ 1 by default; its optional margin override differs from the default projection-label path. Projection flags do not themselves execute the contract-style balance transition. |
| C04 | P0 | **Close factor missing.** `protocol.py:815–820` caps repayment only by debt, then available collateral. The Polygon archived logic and Origin 3.1/3.2 select **50% when HF > 0.95, 100% when HF ≤ 0.95**, subject to liquidation eligibility. At HF **0.98**, debt **86.73469388**, the simulator accepted **80**, whereas those versions permit at most **43.36734694** before token rounding. | Changes debt and collateral after historical liquidations. Version-specific fixtures must cover HF=1, HF=0.95, both sides, collateral exhaustion, and residual rules. The small-dollar probe is deliberately scoped to the earlier rules: later 3.3 sizing can permit full repayment for small positions. |
| C05 | P0 | **Liquidation settlement incomplete.** `protocol.py:823` hardcodes a 5% bonus, omits protocol liquidation fees and `receiveAToken`, and always credits underlying collateral to a liquidator wallet. Historical Polygon configuration includes different bonuses. Contract [LiquidationLogic](https://github.com/aave-dao/aave-v3-origin/blob/v3.1.0/src/core/contracts/protocol/libraries/logic/LiquidationLogic.sol) uses reserve/eMode bonus, protocol fee and collateral delivery mode. | Wrong seized balances, treasury allocation, cash liquidity and post-liquidation positions. Replay verified debt and collateral legs without forcing HF labels; compare event fields and both token flows. `execute_transaction_silent` also funds the liquidator synthetically, so it does not verify liquidator funding/allowance feasibility. |
| C06 | P0 | **No historical risk-parameter fidelity.** `simulator/reserve_config.py` uses “V3-like” defaults and fallback parameters for unknown assets; `setup_protocol` uses synthetic LP deposits. The optional `data/reserves/threshold_history.json` is absent. Historical BGD Polygon report examples below disagree materially. | LT, LTV, available liquidity, utilization/rates and action feasibility are not reconstructed from the historical market. Build a versioned parameter/state series; unavailable coverage must remain explicit. Do not replace all dates with one old snapshot or today's parameters. |
| C07 | P0 | **Collateral flags/eMode/isolation not reconstructed.** `get_user_account_data` counts all stored supply as enabled collateral and uses reserve defaults. User configuration bits, eMode categories, isolation ceilings, pause/freeze, supply/borrow caps, siloed borrowing and related validation are absent from the basic methods. | HF and feasibility can be wrong even with exact balances. Snapshot `COLLATERAL` side means a lending position; it does not prove the user enabled it as collateral. Contract [GenericLogic](https://github.com/aave-dao/aave-v3-origin/blob/v3.1.0/src/core/contracts/protocol/libraries/logic/GenericLogic.sol) reads user configuration/eMode; [ValidationLogic](https://github.com/aave-dao/aave-v3-origin/blob/v3.1.0/src/core/contracts/protocol/libraries/logic/ValidationLogic.sol) applies the other guards. Do not fill historical flags from present-day account state. |
| C08 | P0 | **Accrual model differs from V3.** `Reserve.update_state` applies linear growth to both supply and variable debt; rates are uniform defaults derived from synthetic/account-local utilization. V3 [MathUtils](https://github.com/aave-dao/aave-v3-origin/blob/v3.2.0/src/contracts/protocol/libraries/math/MathUtils.sol) distinguishes linear supply from compounded debt interest, with fixed-point arithmetic and reserve rate/index updates. The replay has one debt balance per symbol, losing stable/variable distinction in earlier deployments. | Balances drift with inactivity. Snapshot indices are retained as provenance but not used to reconstruct intervening accrual. Validate per-market indices/rates, stable debt where applicable, all reserve balances at one timestamp, and rounding near boundaries. |
| C09 | P0 | **Negative warning lead time.** `strategy_3_binary_search`, `tools/liquidation_detection_strategies.py:462–499`, starts one coarse step before the base time if the first check is already positive. An always-positive probe returns **−3544 seconds**. Binary search additionally assumes a monotone predicate, which changing prices need not provide. | Apparent immediate/delayed liquidation timing can be invalid, affecting rapid-liquidation exclusions. Clamp the search domain in a future fix, test already-positive checkpoints, and validate earliest crossing under nonmonotone prices. |
| C10 | P1 | **Token-symbol aggregation loses asset identity.** `utils/indexed_checkpoint.py` explicitly calls `PositionHistory(..., allow_symbol_aggregation=True)`, bypassing the helper's default guard. It can aggregate distinct underlyings with the same symbol, including legacy/native USDC mappings. | A “same-symbol” repayment cap is not necessarily same-underlying-asset funding. Model reserves, prices, wallets and events by chain + asset/market address, with symbols only for display. Unknown mappings currently raise an explicit error, which is preferable to guessing. |
| C11 | P1 | **Stale snapshot balances are installed as current positions.** `PositionHistory.before` carries the last recorded raw balance of each position forward; `run_single_simulation.py:209–220` rescales it using the simulator index at the anchor. Probe: a 100-token debt stays **100 after 30 days** with no intervening index reconstruction. Only aggregate debt is adjusted during anchoring; aggregate supplied liquidity is not reconciled. | Good provenance and strict-before selection do not make a stale balance exact. Reconstruct applicable reserve indices/rates between snapshot and checkpoint and track snapshot age/coverage; stable debt needs its own semantics. Reconcile global reserves separately. |
| C12 | P1 | **Transfers/flash loans are not wired into the active evaluator.** The dispatcher handles only four core actions, Liquidated and PriceUpdate. Neither the current profiles nor the evaluator connect downloaded transfer/flashloan rows to that dispatcher. Unknown actions fall through as success; an isolated `Flashloan` probe returns **True without applying a transaction**. | Download completeness is not simulator utilization. An eventual integration must explicitly reject unsupported actions and record applied/excluded supplementary IDs; never imply downloaded data are all replayed. |
| C13 | P1 | **Timestamp-only/non-atomic replay.** The active runner sorts by timestamp; it does not reconstruct block/transaction/log order, rollback a whole failed transaction, or dispatch beneficiary/payer separately. Failed actions are recorded but execution continues; reserve updates can already have happened before an action fails. | Atomic flash-loan bundles and transfers accompanying core actions cannot safely be appended as independent operations. Use complete receipts/order where available, distinguish user/onBehalfOf/to/liquidator, and mark unverifiable bundles rather than guessing. |
| C14 | P1 | **Historical price/state evidence is not independent chain truth.** `market/market_conditions_simulator.py` builds prices from transaction feature tables (`priceInUSD`, maximum at a shared timestamp). Indexed snapshots are subgraph reconstructions. Neither is a verified oracle round/reserve state at each target block. | Correlations and same-source snapshot agreement cannot certify oracle/HF accuracy. Obtain independently attributable historical oracle/index observations where free access is available, and report staleness/missing coverage. |

### Concrete historical parameter counterexamples

The [BGD pre-upgrade Polygon report](https://github.com/bgd-labs/proposal-3.0.2-upgrade/blob/main/reports/pre-upgrade-polygon.json) supplies these examples. This is evidence that defaults are not universally historical values, **not** a configuration to apply to the entire dataset.

| Asset | Report LTV / LT / bonus | Simulator LTV / LT / bonus |
|---|---|---|
| WMATIC (same underlying now named WPOL) | 65% / 70% / 10% | 80% / 85% / 5% |
| WETH | 80% / 82.5% / 5% | 82.5% / 85% / 5% |
| WBTC | 70% / 75% / 6.5% | 73% / 78% / 5% |
| AAVE | 60% / 70% / 7.5% | 66% / 73% / 5% |

The same report records nonzero liquidation protocol fees and differing reserve factors. Its eMode categories also have different LTV/LT/bonus settings, which cannot be reproduced by the simulator's single default per symbol.

### Dust and version caveat

The examined Polygon archived implementations and Origin 3.1/3.2 validate HF < 1 and selected-asset debt; no general “tiny debt can be liquidated regardless of HF” exception was found. Origin 3.3 adds small-position close-factor and residual-balance rules, but [its validation still requires HF < 1](https://raw.githubusercontent.com/aave-dao/aave-v3-origin/v3.3.0/src/contracts/protocol/libraries/logic/ValidationLogic.sol). These rules are not equivalent to either the simulator's $1 debt/$10 collateral warning or a 1% ratio.

A separate bug exists in the optional debt-to-cover dust path (`validate_simulator.py:190–218`): it multiplies a token amount by a **Reserve object**, catches the resulting exception, and divides token units by USD debt for the ratio. The default projection calls do not pass an estimated debt-to-cover, so this specific ratio bug is not evidence that every current-run label uses that branch.

## How the extra data are actually used

| Downloaded source | Available indexed rows | Active recommendation generation / paired evaluator | Separate diagnostic replay | Assessment |
|---|---:|---|---|---|
| `positionSnapshots` | 13,075,112 | **Used** in indexed variant as a one-time pre-intervention position anchor; core variant does not anchor. | `--variant snapshots` anchors strictly prior-block state at historical checkpoints. | Partial correctness: raw units, identity checks, source provenance and no future counterfactual overwrites are sound. Missing flags, stale balances/indices, incomplete market mapping and symbol aggregation remain serious limitations. |
| `transfers` | 1,561,800 | **Not consumed directly.** Earlier transfers may be reflected indirectly in a later snapshot balance. | `transfer_view` filters to identifiable collateral/aToken transfers, excludes core-overlapping hashes and ambiguous repetitions, and writes exclusion reasons. `apply_observed_transfer` changes supplied-token ownership without funding an underlying wallet or changing reserve cash. | The wallet/cash treatment is appropriate for aToken ownership transfers, but this is not full contract transfer validation or complete history. The historical diagnostic applies a delayed transfer using the next checkpoint timestamp (`enriched_replay.py:70`), rather than its own event timestamp. Core-hash exclusion can omit real extra transfers in a multi-action transaction. |
| `flashloans` | 4,057,345 | **Not consumed.** | No dedicated flash-loan execution in the enriched harness. | Do not blindly turn every flash loan into lasting debt. Actual [FlashLoanLogic](https://github.com/aave-dao/aave-v3-origin/blob/v3.1.0/src/core/contracts/protocol/libraries/logic/FlashLoanLogic.sol) executes a callback, then either repayment plus premium or permitted debt opening. A core Borrow may already encode debt opening; adding it twice is wrong. Premium/rate effects, callback actions and transaction success need explicit accounting. |
| Collateral/eMode/rate-mode changes, reserve parameters, oracle history | Unavailable as candidate history entities in the serving schema | No complete historical reconstruction | Coverage manifest explicitly reports unavailable | Their absence is not zero activity. Latest related entity fields must not be joined as historical truth. |

The [Messari Polygon mapping](https://github.com/messari/subgraphs/blob/2711ac91ef119f321f65b339e10a57f9aa74f9d8/subgraphs/aave-forks/protocols/aave-v3/src/mapping.ts), [position manager](https://github.com/messari/subgraphs/blob/2711ac91ef119f321f65b339e10a57f9aa74f9d8/subgraphs/aave-forks/src/sdk/position.ts) and schema support treating snapshot balances as native token units and retaining indices separately. The schema distinguishes a lending-side position from its `isCollateral` flag; that flag is not included in our snapshot payload. The public mapping also contains transfer amount/version handling and beneficiary-aware supply handlers. **Its exact source-to-deployed-subgraph-IPFS match has not been verified**, so it is supporting semantics evidence, not certification of the serving deployment. Code in a public mapping does not prove a history entity is queryable in the serving schema.

### What is already correct or appropriately conservative

- Strict HF rejection is the default in the actual liquidation method; the running dispatcher does not pass the optional margin override.
- Basic borrow LTV capacity, over-withdrawal, reserve liquidity and same-symbol repayment bounds exist; earlier improvements should be preserved. They do not implement all contract guards.
- Liquidation debt is capped to reconstructed debt and collateral availability, and the borrower/liquidator roles are separate. These partial safeguards do not replace close factors or exact settlement.
- Snapshot raw amounts are divided by decimals once, not blindly multiplied by their RAY index twice. Current/future timestamp snapshots are excluded.
- The recommendation anchor is applied once before intervention and does not repeatedly force counterfactual states back to recorded outcomes. External wallets are preserved when anchoring.
- Supplementary data remain separate from the core survival-event view. This audit found no path that inserts downloaded flash loans/transfers as additional survival targets.

## Consequences for the running study and next steps

The active run is useful as a reproducible **current implementation baseline**. Its “HF-only” subset excludes some dust/threshold labels but can retain adjusted HF and margin-based labels; it is not automatically strict contract eligibility. Its inferred wallets, known future-dependent follow-up horizon, timestamp ordering, fixed historical models and selected cohort remain additional assumptions. Paired simulation does not independently establish causal prevention or real liquidator execution.

After the run finishes, proposed order of work:

1. Correct C01/C02/C09 with focused timestamp/query invariance probes, keeping old results immutable.
2. Separate contract eligibility from warning policies and observed execution; remove zero-debt false liquidation labels from the protocol target without silently excluding reconstruction failures.
3. Implement versioned liquidation limits/settlement and verified historical reserve/user settings. First reconstruct the missing Polygon upgrade/configuration timeline; do not tune thresholds to match observed labels.
4. Use address-based assets and reserve index series; quantify snapshot ages and unknown flags before claiming snapshot state accuracy.
5. Integrate only verified supplementary transaction effects, with receipt order, atomicity and explicit exclusion ledgers. Evaluate core/fixed logic versus enriched/fixed logic on the identical cohort/window.
6. Report absolute balance/HF errors, eligibility confusion matrices, observed-event detection separately, recommendation abstentions/feasibility, and prevention versus delay. Re-run paper-facing results only after these validations.

**No fixes or manuscript changes were made in this audit.** Full historical contract equivalence remains unverified; the defects above are sufficient to reject a claim of equivalence now.
