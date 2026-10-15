"""Account-clustered intervals for saved, known-label seven-day predictions.

This does not retrain models or turn the known-label subset into a population
survival estimate. All resamples keep an account's test observations together.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame):
    labels = frame["label"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)
    if len(set(labels)) != 2:
        return None
    return dict(roc_auc=float(roc_auc_score(labels, probability)),
                average_precision=float(average_precision_score(labels, probability)))


def audit(path, repetitions, seed):
    frame = pd.read_csv(path)
    if frame.case_id.duplicated().any():
        raise ValueError(f"Duplicate test case IDs in {path}")
    if frame.horizon.nunique() != 1 or int(frame.horizon.iloc[0]) != 604800:
        raise ValueError("Expected saved seven-day predictions")
    known = frame[(frame.event.astype(bool)) |
                  (frame.duration >= frame.horizon)].copy()
    known["label"] = (known.event.astype(bool) &
                      (known.duration <= known.horizon)).astype(int)
    accounts = sorted(known.user.unique())
    groups = {account: block for account, block in known.groupby("user", sort=False)}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        selected = rng.choice(accounts, size=len(accounts), replace=True)
        result = metrics(pd.concat([groups[account] for account in selected],
                                   ignore_index=True))
        if result is not None:
            draws.append(result)
    point = metrics(known)
    intervals = {key: [float(np.quantile([row[key] for row in draws], q))
                       for q in (0.025, 0.975)] for key in ("roc_auc", "average_precision")} if draws else None
    return dict(source=str(path.resolve()), source_sha256=file_hash(path),
                saved_test_rows=len(frame), known_horizon_rows=len(known),
                excluded_short_followup=len(frame)-len(known),
                unique_test_accounts=frame.user.nunique(),
                unique_known_label_accounts=len(accounts),
                accounts_with_repeated_known_rows=int((known.groupby("user").size() > 1).sum()),
                positive_rows=int(known.label.sum()),
                positive_accounts=known.loc[known.label.eq(1), "user"].nunique(),
                point_estimate=point, cluster_bootstrap_95_percentile=intervals,
                bootstrap_valid_draws=len(draws), bootstrap_requested=repetitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.predictions.glob("*-XGBoostCox-predictions.csv"))
    if not files:
        raise ValueError("No saved XGBoost-Cox predictions found")
    results = {path.name.split("-")[0]: audit(path, args.bootstrap, args.seed)
               for path in files}
    report = dict(tasks=results, method="Account-cluster bootstrap of saved known-seven-day-label rows; no refit or censoring weighting", code_sha256=file_hash(__file__))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({task: {key: row[key] for key in
                             ("known_horizon_rows", "unique_known_label_accounts",
                              "positive_rows", "positive_accounts", "bootstrap_valid_draws")}
                      for task, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
