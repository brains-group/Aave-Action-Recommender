"""Read-only progress summary for a shortest-liquidation study run."""

import argparse
import json
import pickle
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def eta(progress, completed_key, remaining_key, elapsed_key):
    if not progress:
        return "?"
    completed = progress.get(completed_key, 0)
    remaining = progress.get(remaining_key, 0)
    elapsed = progress.get(elapsed_key, 0)
    if not completed or not isinstance(remaining, (int, float)):
        return "?"
    seconds = max(0, round(elapsed * remaining / completed))
    return f"about {seconds // 3600}h {(seconds % 3600) // 60}m at the current rate"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    root = a.run
    status = read(root / "status.json") or {}
    study_key = status.get("study_key")
    study_root = root / "studies" / study_key if study_key else root
    cohort = read(root / "cohort" / "manifest.json") or {}
    print("cohort:", cohort.get("selected", "pending"), "rows;", cohort.get("distinct_accounts", "?"), "accounts")
    parts = len(list((root / "histories").glob("*.pkl")))
    print("histories:", f"{parts}/25", "complete" if (root / "histories" / "complete.json").exists() else "running/pending")
    stages = (
        ("core-state generation", study_root / ("generation-core" if study_key else "generation-core-quiet")),
        ("agent evaluation", study_root / "agent-evaluation"),
        ("policy baselines", study_root / "policy-baselines"),
    )
    for label, folder in stages:
        progress = read(folder / "progress.json")
        completed = read(folder / "complete.json")
        pass_finished = read(folder / "pass-complete.json")
        if label == "core-state generation":
            saved = len(list((folder / "cases").glob("*.pkl")))
            total = cohort.get("selected", "?")
            failures = 0
            for case in (folder / "cases").glob("*.pkl"):
                try:
                    with case.open("rb") as stream:
                        failures += bool(pickle.load(stream).get("error"))
                except (OSError, EOFError, pickle.UnpicklingError):
                    failures += 1
            print(label + ":", f"{saved}/{total} saved, {failures} errors",
                  "ETA", eta(progress, "done_this_run", "remaining_this_run", "elapsed_seconds"))
            if completed and completed.get("cases", 0) == total:
                print("  generation complete:", completed)
        elif label == "agent evaluation":
            saved = len(list((folder / "cases").glob("*.json")))
            print(label + ":", f"{saved}/{2*cohort.get('selected', 0)} case-variants saved",
                  "ETA", eta(progress, "completed_this_run", "remaining_this_run", "seconds"))
            if completed:
                print("  evaluation complete:", completed)
        elif label == "policy baselines":
            saved = len(list((folder / "cases").glob("*/complete.json")))
            errors = len(list((folder / "cases").glob("*/error.json")))
            unavailable = len(list((folder / "cases").glob("*/unavailable.json")))
            print(label + ":", f"{saved} complete, {errors} errors, {unavailable} unavailable")
            if pass_finished:
                print("  first pass ended; inspect errors/unavailable:", pass_finished)
        elif completed:
            print(label + ":", "complete", completed)
        elif progress:
            done = progress.get("done_this_run", progress.get("completed_this_run", progress.get("completed", "?")))
            remaining = progress.get("remaining_this_run", progress.get("unavailable", "?"))
            print(label + ":", "done", done, "remaining", remaining, "elapsed seconds", progress.get("elapsed_seconds", progress.get("seconds", "?")))
        else:
            print(label + ": pending")
    print("runner status:", status or "not started")


if __name__ == "__main__":
    main()
