"""Train only missing scheduled models needed by the shortest-time cohort.

Existing fitted artifacts are linked read-only into an isolated model cache.
The normal model-training code filters training outcomes to the model date.
"""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd


EVENTS = ("Borrow", "Deposit", "Repay", "Withdraw", "Liquidated")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def demand(cohort_path, histories, date_ranges):
    cohort = pd.read_csv(cohort_path)
    train_dates, test_dates = pickle.load(open(date_ranges, "rb"))
    dates = train_dates.union(test_dates)
    cases = {}
    by_user = {}
    for index, row in cohort.iterrows():
        date = dates[dates <= pd.to_datetime(row.timestamp, unit="s")].max()
        if pd.isna(date):
            raise ValueError(f"No model date before case {index}")
        cases[index] = dict(
            user=row.user, timestamp=row.timestamp, model_date=date,
            origins={str(row["Index Event"]).title()},
        )
        by_user.setdefault(row.user, []).append(index)
    for source in sorted(histories.glob("*.pkl")):
        frame = pickle.load(source.open("rb"))
        if frame.empty:
            continue
        for user, rows in frame.groupby("user"):
            for index in by_user.get(user, ()):
                case = cases[index]
                past = rows.loc[rows.timestamp < case["timestamp"], "Index Event"]
                case["origins"].update(str(value).title() for value in past.unique())
    requested = {}
    for index, case in cases.items():
        for origin in case["origins"]:
            for outcome in EVENTS:
                if (origin, outcome) == ("Liquidated", "Liquidated"):
                    continue
                key = (str(case["model_date"]), origin, outcome)
                requested.setdefault(key, []).append(index)
    return requested, len(cohort)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    source = a.source_cache.resolve()
    output = a.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("The isolated model cache must not overwrite the source cache")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("models", "data"):
        (output / name).mkdir(exist_ok=True)
        for artifact in (source / name).iterdir():
            link = output / name / artifact.name
            if artifact.is_file() and not link.exists():
                link.symlink_to(artifact.resolve())
    date_ranges = output / "date_ranges.pkl"
    if not date_ranges.exists():
        date_ranges.symlink_to((source / "date_ranges.pkl").resolve())
    requested, cases = demand(a.cohort, a.histories, date_ranges)
    missing = []
    for (date, origin, outcome), indices in sorted(requested.items()):
        stem = f"xgboost_cox_{origin}_{outcome}_{date.replace(' ', '_')}"
        model = source / "models" / f"{stem}.pkl"
        baseline = source / "models" / f"{stem}_baseline.pkl"
        if not model.is_file() or not baseline.is_file():
            missing.append(dict(date=date, origin=origin, outcome=outcome, cases=indices))
    identity = dict(
        cohort_sha256=sha(a.cohort),
        history_manifest_sha256=sha(a.histories / "manifest.json"),
        date_ranges_sha256=sha(date_ranges),
        source_cache=str(source),
        source_model_count=len(list((source / "models").glob("*.pkl"))),
        source_data_count=len(list((source / "data").glob("*.pkl"))),
        script_sha256=sha(__file__),
        requested_model_count=len(requested),
        affected_cases=len(set(index for item in missing for index in item["cases"])),
        initially_missing=missing,
        cohort_cases=cases,
    )
    manifest = output / "training_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Training inputs changed; choose a fresh model cache")
    save(manifest, identity)
    print(f"Missing {len(missing)} scheduled models for {identity['affected_cases']} cases", flush=True)

    os.environ["AAVE_EVALUATION_CACHE"] = str(output)
    os.environ.pop("AAVE_FROZEN_MODELS", None)
    os.environ.pop("AAVE_SURVIVAL_DATA", None)
    from utils.model_training import get_model_for_pair_and_date

    def available(item):
        stem = f"xgboost_cox_{item['origin']}_{item['outcome']}_{item['date'].replace(' ', '_')}"
        return all((output / "models" / f"{stem}{suffix}.pkl").is_file()
                   for suffix in ("", "_baseline"))

    todo = [item for item in missing if not available(item)]
    if a.limit is not None:
        todo = todo[: a.limit]
    for number, item in enumerate(todo, 1):
        date = pd.Timestamp(item["date"])
        print(f"Training {number}/{len(todo)}: {item['origin']}->{item['outcome']} at {date}", flush=True)
        result = get_model_for_pair_and_date(
            item["origin"], item["outcome"], model_date=date, verbose=False
        )
        if result is None:
            raise RuntimeError(f"No training data for {item}")
        save(output / "training_progress.json", dict(completed=number, total=len(todo), last=item))
    if all(available(item) for item in missing):
        save(output / "complete.json", dict(trained=len(missing), requested=len(requested), cases=cases))


if __name__ == "__main__":
    main()
