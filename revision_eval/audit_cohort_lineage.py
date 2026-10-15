"""Describe saved original and separate shortest-time cohort identities.

This is an inventory, not a claim that the manuscript's 8,400 intermediate
checkpoints have been recovered. It reads both inputs and the original aggregate
statistics without modifying them.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_stats(source):
    overall = json.loads(source.read_text())["overall"]
    return {key: value for key, value in overall.items()
            if isinstance(value, (int, float, str, bool)) or value is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-set", type=Path, required=True)
    parser.add_argument("--high-risk", type=Path, required=True)
    parser.add_argument("--original-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_set, usecols=["user", "timestamp", "pool",
                                                   "Index Event", "Outcome Event",
                                                   "timeDiff", "status"])
    stress = pd.read_csv(args.high_risk, usecols=["user", "timestamp", "pool",
                                                     "Index Event", "Outcome Event",
                                                     "timeDiff", "status"])
    key_columns = ["user", "timestamp", "pool", "Index Event"]
    original_keys = set(map(tuple, train[key_columns].itertuples(index=False, name=None)))
    stress_keys = set(map(tuple, stress[key_columns].itertuples(index=False, name=None)))
    pair_counts = (train.groupby(["Index Event", "Outcome Event"], dropna=False)
                   .size().sort_values(ascending=False))
    high_risk_pairs = (stress.groupby(["Index Event", "Outcome Event"], dropna=False)
                       .size().sort_values(ascending=False))
    stats = scalar_stats(args.original_statistics)
    report = dict(sources={str(p.resolve()): file_hash(p) for p in
                           (args.train_set, args.high_risk, args.original_statistics)},
                  original_cached_selection=dict(rows=len(train), accounts=train.user.nunique(),
                                                 unique_user_timestamp=len(train.drop_duplicates(["user", "timestamp"])),
                                                 unique_full_keys=len(original_keys),
                                                 pair_counts={f"{a} -> {b}": int(count)
                                                              for (a, b), count in pair_counts.items()},
                                                 observed_liquidation_outcomes=int(((train["Outcome Event"].str.lower() == "liquidated") &
                                                                                     (train.status == 1)).sum())),
                  separate_shortest_time_test=dict(rows=len(stress), accounts=stress.user.nunique(),
                                                   unique_full_keys=len(stress_keys),
                                                   pair_counts={f"{a} -> {b}": int(count)
                                                                for (a, b), count in high_risk_pairs.items()},
                                                   exact_key_overlap_with_original=len(original_keys & stress_keys)),
                  original_paper_aggregate={key: stats.get(key) for key in
                                            ("processed", "liquidated_without", "liquidated_with",
                                             "improved", "worsened", "delayed_liquidations_7d")},
                  unresolved="The submitted-paper 8,400 intermediate checkpoint IDs and case-level filter ledger are not established by these three inputs; do not equate the 12,000 cached selection with that intermediate count.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"original_cached_rows": len(train),
                      "original_accounts": train.user.nunique(),
                      "shortest_time_rows": len(stress),
                      "exact_key_overlap": len(original_keys & stress_keys),
                      "paper_processed": stats.get("processed")}, indent=2))


if __name__ == "__main__":
    main()
