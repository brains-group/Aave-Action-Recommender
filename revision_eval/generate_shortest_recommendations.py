"""Generate agent recommendations for a frozen shortest-liquidation cohort.

Uses the existing fitted models and current actionAgentTraining.recommend_action.
Outputs are isolated and resumable; no models are trained or data collected.
"""

import argparse
import contextlib
import json
import os
import pickle
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .prepare_shortest_histories import EVENTS, digest
from .followup import fixed_followup


def select_saved_cases(cohort, case_ids):
    """Select existing IDs without resampling or changing their source order."""
    if case_ids is None:
        return cohort
    if not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError('Case IDs must be a nonempty list without duplicates')
    source = cohort['case_id'].astype(str)
    if source.duplicated().any() or set(case_ids) - set(source):
        raise ValueError('Case IDs must match unique saved cohort checkpoints')
    return cohort.loc[source.isin(case_ids)].copy()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=4)
    tmp.replace(path)


def history(user_id, up_to_timestamp):
    frame = HISTORIES.get(user_id)
    if frame is None:
        return pd.DataFrame()
    return frame.loc[frame.timestamp <= up_to_timestamp].sort_values("timestamp")


def generate(index):
    import actionAgentTraining as agent

    row = ROWS[index]
    case_id = str(row["case_id"])
    row = row.drop(labels=["stress_origin", "source_row", "case_id"], errors="ignore")
    dest = RUN / "cases" / f"{index:05d}.pkl"
    started = time.time()
    # The simulator prints every sampled timestamp on stdout; on a long replay
    # this can exceed tens of MB per case without adding useful diagnostics.
    # Preserve stderr and the serialized traceback, but discard that raw dump.
    with Path(os.devnull).open("w") as discard, (RUN / "logs" / f"worker-{os.getpid()}.log").open("a") as log:
        with contextlib.redirect_stdout(discard), contextlib.redirect_stderr(log):
            try:
                rec, info = agent.recommend_action(row)
                rec["case_id"] = case_id
                result = dict(index=index, case_id=case_id, recommendation=(rec, info), error=None)
            except Exception:
                rec = agent.generate_next_transaction(row, "Deposit", amount=0)
                rec["case_id"] = case_id
                rec["generation_error"] = "High-risk recommendation generation failed"
                result = dict(index=index, case_id=case_id, recommendation=(rec, {}), error=traceback.format_exc())
            result["seconds"] = time.time() - started
            save(dest, result)
    return {"index": index, "seconds": round(result["seconds"], 2), "error": bool(result["error"])}


def main():
    global RUN, ROWS, HISTORIES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True)
    p.add_argument("--model-cache", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--checkpoint-mode", choices=("core", "indexed"), default="core")
    p.add_argument("--horizon-seconds", type=int)
    p.add_argument("--case-ids", type=Path, help="JSON list of existing checkpoint IDs; no new sampling")
    p.add_argument("--read-cache-roots", type=Path, help="JSON list of read-only compatible simulation cache roots")
    a = p.parse_args()
    if a.horizon_seconds is not None and a.horizon_seconds <= 0:
        raise ValueError("--horizon-seconds must be positive")
    RUN = a.run.resolve()
    for name in ("cases", "logs", "cache"):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    if not (a.histories / "complete.json").is_file():
        raise ValueError("Historical feature extraction is incomplete")
    identity = dict(
        cohort_sha256=digest(a.cohort),
        history_manifest_sha256=digest(a.histories / "manifest.json"),
        coverage_sha256=digest(a.coverage),
        assets_sha256=digest(a.assets),
        model_cache=str(a.model_cache.resolve()),
        model_listing={
            f.name: [f.stat().st_size, f.stat().st_mtime_ns]
            for f in (a.model_cache / "models").iterdir() if f.is_file()
        },
        code_sha256=digest(Path(__file__)),
        agent_sha256=digest(Path(__file__).resolve().parents[1] / "actionAgentTraining.py"),
        followup_sha256=digest(Path(__file__).with_name("followup.py")),
        checkpoint_mode=a.checkpoint_mode,
        horizon_seconds=a.horizon_seconds,
    )
    if a.case_ids is not None:
        identity['case_ids_sha256'] = digest(a.case_ids)
    if a.read_cache_roots is not None:
        identity['read_cache_roots_sha256'] = digest(a.read_cache_roots)
    manifest = RUN / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Generation inputs changed; use a fresh run directory")
    else:
        manifest.write_text(json.dumps(identity, indent=2))
    for name in ("models", "data", "date_ranges.pkl"):
        dest = RUN / "cache" / name
        if not dest.exists():
            dest.symlink_to((a.model_cache / name).resolve(), target_is_directory=name != "date_ranges.pkl")
    empty = RUN / "empty-recommendations.pkl"
    if not empty.exists():
        save(empty, {})
    os.environ.update(
        AAVE_EVALUATION_CACHE=str(RUN / "cache"),
        AAVE_RECOMMENDATIONS_FILE=str(empty),
        AAVE_SIMULATION_CACHE=str(RUN / "simulation-cache"),
        AAVE_FROZEN_MODELS="1",
    )
    for key in ("AAVE_SURVIVAL_DATA", "AAVE_CORE_TRANSACTIONS"):
        os.environ.pop(key, None)
    if a.checkpoint_mode == "indexed":
        os.environ.update(AAVE_INDEXED_COVERAGE=str(a.coverage.resolve()), AAVE_INDEXED_ASSETS=str(a.assets.resolve()))
    else:
        os.environ.pop("AAVE_INDEXED_COVERAGE", None)
        os.environ.pop("AAVE_INDEXED_ASSETS", None)
    import actionAgentTraining as agent
    if a.read_cache_roots is not None:
        from . import audit_baseline_tuning as audit
        from utils.simulations import simulation_code_identity
        cache_roots = [Path(p) for p in json.loads(a.read_cache_roots.read_text())]
        code_identity = simulation_code_identity()
        original = agent.get_simulation_outcome
        def cached_outcome(rec, suffix, **kwargs):
            hit = audit.lookup(rec, suffix, kwargs, code_identity, cache_roots)
            if hit:
                with Path(hit).open('rb') as stream:
                    return pickle.load(stream)
            return original(rec, suffix, **kwargs)
        agent.get_simulation_outcome = cached_outcome
    if a.horizon_seconds is not None:
        agent.get_limited_user_profile = fixed_followup(
            agent.get_limited_user_profile, a.horizon_seconds
        )

    cohort = pd.read_csv(a.cohort)
    cohort = select_saved_cases(cohort, json.loads(a.case_ids.read_text()) if a.case_ids else None)
    if a.limit is not None:
        if a.limit <= 0:
            raise ValueError("--limit must be positive")
        cohort = cohort.head(a.limit)
    ROWS = [row for _, row in cohort.iterrows()]
    parts = []
    for ie in EVENTS:
        for oe in EVENTS:
            with (a.histories / f"{ie}-{oe}.pkl").open("rb") as f:
                frame = pickle.load(f)
            if not frame.empty:
                parts.append(frame)
    all_history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    HISTORIES = {user: frame for user, frame in all_history.groupby("user", sort=False)} if not all_history.empty else {}
    agent.get_user_history = history
    from simulator import utils as simulator_utils
    simulator_utils.setup_protocol()
    tasks = [i for i in range(len(ROWS)) if not (RUN / "cases" / f"{i:05d}.pkl").exists()]
    start = time.time()
    with Pool(a.workers) as pool:
        for done, status in enumerate(pool.imap_unordered(generate, tasks), 1):
            progress = dict(done_this_run=done, remaining_this_run=len(tasks)-done, total=len(ROWS), elapsed_seconds=round(time.time()-start), last=status)
            (RUN / "progress.json").write_text(json.dumps(progress))
            if done % 10 == 0 or status["error"]:
                print(progress, flush=True)
    records = []
    for i in range(len(ROWS)):
        with (RUN / "cases" / f"{i:05d}.pkl").open("rb") as f:
            records.append(pickle.load(f))
    save(RUN / "recommendations.pkl", {r["case_id"]: r["recommendation"] for r in records})
    (RUN / "complete.json").write_text(json.dumps(dict(cases=len(records), errors=sum(bool(r["error"]) for r in records)), indent=2))
    print((RUN / "complete.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
