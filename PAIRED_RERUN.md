# Paired recommendation rerun — 2026-09-10

The actual performSimulations.process_recommendation path is evaluated for all 12,000
saved recommendations, under current-core and indexed-checkpoint variants. Original
recommendations, paper-era simulation caches, statistics and manuscript assets are preserved.
The full run has its own output/cache directory and writes one JSON per recommendation/variant.
Do not report a prevention rate until completion and report matched denominators/exclusions.

Changes:
- AAVE_EVALUATION_CACHE isolates simulation/statistics caches; AAVE_RECOMMENDATIONS_FILE
  selects a frozen existing recommendation file. No models or recommendations are retrained.
- Cache identity hashes simulator Python sources, consumer utils and price history, plus
  simulation arguments. Resume identity also checks driver code, recommendations, coverage,
  asset mappings and profile path/size/mtime metadata. A changed input requires a new run.
- Optional AAVE_INDEXED_COVERAGE and AAVE_INDEXED_ASSETS attach strictly earlier timestamp
  position snapshots before the last historical transaction timestamp. Both arms receive the
  same anchor and replay the same last historical actions. Wallet funding remains unchanged.
  The recommendation then modifies the anchored state; no future snapshot resets occur.
- Last indexed balances are carried forward, not exact historical accrual. Current legacy
  symbol-based asset aggregation remains explicit. Missing snapshot history or asset mappings
  produces an exclusion/error, not an empty position or silent fallback.
- Simulator template cloning shares its read-only price-history mapping while copying mutable
  prices/reserves/accounts. Runtime methods read the history and do not modify it; callers
  must replace a history with set_price_history rather than mutate the shared mapping.
- Transaction execution failures and checkpoint provenance are retained in cached results.

Existing performSimulations behavior is preserved: inferred profile capital, repayment
feasibility checks, variable horizons, dust/rapid-liquidation/zero-debt filters, and HF-only
filtering based on either arm. Later recorded transactions determine horizon and labels;
they are not actually replayed by this script. Four strategies are active in the existing
run_all_strategies implementation, despite six-strategy prose/comments. These assumptions
make prevention conditional on this simulation policy, not a real-world causal estimate.
The arms retain the existing script's different time origins/horizon adjustment; this rerun
is a behavior-preserving comparison, not a correction of every methodological issue.

20 simulator tests pass. Tests verify a single anchor preserves wallet accounting and the
intervention delta, and that shared history does not share mutable account/reserve state.
A 16-case pilot (eight recommendations in both modes) completed; it is a smoke/timing check,
not a reported prevention estimate. The mean/median cost varies substantially by profile.

Full output: /home/spadef/data/craft-soc/data/evaluation/recommendations-20260910/
Use launcher receipt.json, console.log, progress.json and complete.json for status.
The run uses eight local workers, no R, no API calls, and no training.

From Aave-Action-Recommender:
```bash
python evaluate_recommendations.py \
  --run-dir /home/spadef/data/craft-soc/data/evaluation/recommendations-20260910 \
  --recommendations /home/spadef/data/craft-soc/data/evaluation/recommendations-20260910/inputs/recommendations.pkl \
  --coverage /home/spadef/data/craft-soc/data/supplementary/aave_v3/polygon/5a19507ddc6a0f4b/coverage.json \
  --assets /home/spadef/data/craft-soc/data/evaluation/recommendations-20260910/inputs/asset-map.json \
  --workers 8
python summarize_recommendation_evaluation.py \
  /home/spadef/data/craft-soc/data/evaluation/recommendations-20260910
```
Do not start a second writer while the recorded job is running. On interruption, the same
command resumes completed case files only when the manifest matches. Cached simulator arms
also resume only with matching code/input identities. The summary marks incomplete runs,
includes excluded cases, paper-style filtered and HF-only totals, unfiltered paired totals,
and a common-case comparison across both variants. The current saved recommendations are
12,000 entries; this should not silently be called the exact paper's 4,882-profile cohort.
No manuscript result has been changed. Final evaluation figures remain pending the full run.
