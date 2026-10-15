"""Audit the original checkpoint trend rule from frozen prediction histories.

This reads the September 2026 refreshed recommendation cache without changing it.
Each result is keyed by the original train-set row number and saved incrementally.
The finite grid is a discrete sensitivity check, not a continuous optimum.
"""

import argparse
import hashlib
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .trend_grid import polynomial_scores


EVENTS = ("Borrow", "Deposit", "Repay", "Withdraw", "Liquidated")
BASE = (0.8, 0.6, 0.3)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_definition():
    grid = np.asarray(list(product(
        np.linspace(0.6, 1.0, 21),
        np.linspace(0.4, 0.8, 21),
        np.linspace(0.1, 0.5, 21),
    )))
    distance = np.linalg.norm(grid - np.asarray(BASE), axis=1)
    return grid, distance


def cached_history(row, cache):
    digest = hashlib.sha256(pickle.dumps(row.to_dict(), protocol=4)).hexdigest()
    path = cache / (digest + ".pkl")
    if not path.is_file():
        return digest, None
    with path.open("rb") as source:
        return digest, pickle.load(source)


def evaluate(history, grid, distances):
    from actionAgentTraining import calculate_trend_slope

    recent = history[max(history)]
    if not all(event in recent and recent[event] is not None and
               np.isfinite(recent[event]) for event in EVENTS):
        raise ValueError("Incomplete latest event predictions")
    immediate = recent["Liquidated"] <= min(recent.values())
    if immediate:
        flags = np.ones(len(grid), dtype=bool)
        baseline = True
    else:
        scores = []
        baseline_scores = []
        for event in EVENTS:
            series = {time: values[event] for time, values in history.items()
                      if values and event in values}
            score = lambda w, l, g: calculate_trend_slope(series, w, l, g)
            scores.append(polynomial_scores(score, grid))
            baseline_scores.append(score(*BASE))
        scores = np.asarray(scores)
        liquidated = EVENTS.index("Liquidated")
        flags = ((scores[liquidated] < 0) &
                 (scores[liquidated] <= scores.min(axis=0)))
        baseline = (baseline_scores[liquidated] < 0 and
                    baseline_scores[liquidated] <= min(baseline_scores))
        # The polynomial expansion must reproduce the direct implementation.
        base_index = np.flatnonzero(np.all(np.isclose(grid, BASE), axis=1))
        if len(base_index) != 1 or bool(flags[base_index[0]]) != bool(baseline):
            raise AssertionError("Grid and original decision disagree at baseline")
    flips = flags != baseline
    minimum = float(distances[flips].min()) if np.any(flips) else None
    grouped = np.round(distances, 4)
    group_flips = [bool(np.any(flips[grouped == distance]))
                   for distance in np.unique(grouped) if distance > 0]
    nonmonotonic = any(near and not far for near, far in
                       zip(group_flips, group_flips[1:]))
    maximum = float(distances.max())
    return dict(base_prediction=bool(baseline), immediate=bool(immediate),
                change_found=bool(np.any(flips)), min_distance=minimum,
                stability_component=minimum / maximum if minimum is not None else 1.0,
                nonmonotonic_group_predicate=bool(nonmonotonic),
                grid_points=len(grid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-set", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--legacy", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid, distances = grid_definition()
    manifest = dict(train_set=str(args.train_set.resolve()),
                    train_set_sha256=file_hash(args.train_set),
                    prediction_cache=str(args.prediction_cache.resolve()),
                    code_sha256=file_hash(__file__), limit=args.limit,
                    legacy=str(args.legacy.resolve()) if args.legacy else None,
                    grid_points=len(grid), coefficients=list(BASE))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Inputs or code changed; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    frame = pd.read_csv(args.train_set)
    if args.limit is not None:
        frame = frame.head(args.limit)
    rows_path = args.output / "rows.jsonl"
    completed = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            record = json.loads(line)
            if record["index"] in completed:
                raise ValueError("Duplicate saved row index")
            completed[record["index"]] = record
    with rows_path.open("a") as destination:
        for index, row in frame.iterrows():
            if int(index) in completed:
                continue
            digest, history = cached_history(row, args.prediction_cache)
            record = dict(index=int(index), user=row["user"],
                          timestamp=float(row["timestamp"]), cache_digest=digest)
            try:
                if history is None:
                    raise FileNotFoundError("Frozen prediction history unavailable")
                record.update(evaluate(history, grid, distances))
            except Exception as error:
                record["error"] = str(error)
            destination.write(json.dumps(record) + "\n")
            destination.flush()
            completed[int(index)] = record
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(frame)} saved", flush=True)
    records = [completed[int(index)] for index in frame.index]
    scored = [row for row in records if "stability_component" in row]
    legacy_comparison = None
    if args.legacy:
        old = pd.read_pickle(args.legacy)
        pairs = old.groupby(["user", "timestamp"], dropna=False)
        compared = changed = 0
        for row in scored:
            try:
                match = pairs.get_group((row["user"], row["timestamp"]))
            except KeyError:
                continue
            if len(match) != 1:
                continue
            compared += 1
            changed += bool(match.iloc[0]["change_found"]) != row["change_found"]
        legacy_comparison = dict(matched_unique_cases=compared,
                                 changed_flip_detection=changed,
                                 note="Prior binary-search cache is historical; model/cache identities may differ")
    summary = dict(selected=len(frame), evaluated=len(scored), errors=len(records)-len(scored),
                   risk_flags=sum(row["base_prediction"] for row in scored),
                   changed_decisions=sum(row["change_found"] for row in scored),
                   nonmonotonic_group_cases=sum(row["nonmonotonic_group_predicate"] for row in scored),
                   stability_score=float(np.mean([row["stability_component"] for row in scored])) if scored else None,
                   legacy_comparison=legacy_comparison,
                   scope="Original cached 12,000 checkpoint rows; refreshed September 2026 predictions; discrete 21^3 grid; not the submitted-paper model cache")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
