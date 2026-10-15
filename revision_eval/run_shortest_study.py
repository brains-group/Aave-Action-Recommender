"""Run the existing evaluator/baseline code with one recorded study follow-up.

The standard drivers and simulator source remain byte-for-byte unchanged so
their existing resumable runs and cache identities are unaffected.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

from .followup import fixed_followup


SOURCE = Path(__file__).resolve().parents[1]


def options(arguments):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--horizon-seconds", type=int)
    selected, remaining = p.parse_known_args(arguments)
    if selected.horizon_seconds is not None and selected.horizon_seconds <= 0:
        raise ValueError("--horizon-seconds must be positive")
    return selected.horizon_seconds, remaining


def read_paths(mode, arguments):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--recommendations", type=Path, required=True)
    if mode == "evaluate":
        p.add_argument("--run-dir", type=Path, required=True)
    else:
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--parent-run", type=Path, required=True)
    value, _unused = p.parse_known_args(arguments)
    return value


def absolute_driver_paths(arguments):
    path_options = {
        "--source", "--parent-run", "--recommendations", "--coverage",
        "--assets", "--policies", "--run", "--run-dir",
    }
    normalized = list(arguments)
    for index, value in enumerate(normalized[:-1]):
        if value in path_options:
            normalized[index + 1] = str(Path(normalized[index + 1]).resolve())
    return normalized


def guard_followup(run, horizon_seconds):
    run.mkdir(parents=True, exist_ok=True)
    dest = run / "followup.json"
    identity = {
        "horizon_seconds_from_recommendation": horizon_seconds,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "followup_sha256": hashlib.sha256(
            Path(__file__).with_name("followup.py").read_bytes()
        ).hexdigest(),
    }
    if dest.exists():
        if json.loads(dest.read_text()) != identity:
            raise ValueError("Follow-up setting changed; use a fresh run directory")
    else:
        if (run / "manifest.json").exists():
            raise ValueError("Existing evaluator run lacks a follow-up identity")
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(identity, indent=2))
        tmp.replace(dest)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"evaluate", "baselines"}:
        raise SystemExit(
            "Usage: python -m revision_eval.run_shortest_study "
            "{evaluate|baselines} [driver arguments] [--horizon-seconds N]"
        )
    mode = sys.argv[1]
    horizon_seconds, original_args = options(sys.argv[2:])
    original_args = absolute_driver_paths(original_args)
    args = read_paths(mode, original_args)
    run = (args.run_dir if mode == "evaluate" else args.run).resolve()
    if mode == "baselines":
        parent = args.parent_run.resolve()
        parent_followup = json.loads((parent / "followup.json").read_text())
        if parent_followup["horizon_seconds_from_recommendation"] != horizon_seconds:
            raise ValueError("Baseline follow-up differs from parent evaluator")
    guard_followup(run, horizon_seconds)

    if mode == "evaluate":
        os.environ.update(
            AAVE_EVALUATION_CACHE=str(run / "cache"),
            AAVE_RECOMMENDATIONS_FILE=str(args.recommendations.resolve()),
        )
    else:
        source = args.source.resolve()
        for name in ("data", "profiles"):
            link = run / name
            if not link.exists():
                link.symlink_to((source / name).resolve(), target_is_directory=True)
        os.chdir(run)
        os.environ.update(
            AAVE_EVALUATION_CACHE=str(run / "cache"),
            AAVE_RECOMMENDATIONS_FILE=str(args.recommendations.resolve()),
            AAVE_SIMULATION_CACHE=str(run / "simulations"),
        )
        for name in (
            "AAVE_SURVIVAL_DATA", "AAVE_CORE_TRANSACTIONS",
            "AAVE_HISTORICAL_EVIDENCE", "AAVE_INDEXED_COVERAGE", "AAVE_INDEXED_ASSETS",
        ):
            os.environ.pop(name, None)
        sys.path.insert(0, str(source))

    import performSimulations as evaluator
    if horizon_seconds is not None:
        evaluator.get_limited_user_profile = fixed_followup(
            evaluator.get_limited_user_profile, horizon_seconds
        )
        if mode == "baselines":
            import utils.simulations as simulations
            simulations.get_limited_user_profile = fixed_followup(
                simulations.get_limited_user_profile, horizon_seconds
            )

    driver = SOURCE / (
        "evaluate_recommendations.py" if mode == "evaluate"
        else "revision_eval/run_existing_baselines.py"
    )
    sys.argv = [str(driver), *original_args]
    module_name = "_shortest_" + mode + "_driver"
    spec = importlib.util.spec_from_file_location(module_name, driver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # Simulator timestamp prints can produce tens of MB for one case. Preserve
    # stderr and structured case errors without changing the frozen drivers.
    with open(os.devnull, "w") as discard:
        module.contextlib = types.SimpleNamespace(
            redirect_stdout=lambda _log: contextlib.redirect_stdout(discard),
            redirect_stderr=contextlib.redirect_stderr,
        )
        module.main()


if __name__ == "__main__":
    main()
