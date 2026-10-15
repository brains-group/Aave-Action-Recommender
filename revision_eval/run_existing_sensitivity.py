"""Bounded, isolated simulator sensitivity on frozen existing recommendations.

Each scenario has its own cache namespace and uses the same case identities,
30-day follow-up, prices and recommendation. Warning-policy results are warning
diagnostics, not observed on-chain liquidation transactions. No source cache is
modified. The reference here is a new fixed-window run, not the paper's result.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path


SCENARIOS = {
    "reference": dict(wallet_multiplier=1.0, policy="protocol", margin=1.0),
    "wallet_one_times": dict(wallet_multiplier=2.0 / 3.0, policy="protocol", margin=1.0),
    "wallet_two_times": dict(wallet_multiplier=4.0 / 3.0, policy="protocol", margin=1.0),
    "no_inferred_wallet": dict(wallet_multiplier=0.0, policy="protocol", margin=1.0),
    "prefix_reinferred_wallet": dict(wallet_multiplier=1.0, policy="protocol", margin=1.0,
                                      wallet_mode="prefix_hybrid", safety_factor=1.5),
    "prefix_one_times": dict(wallet_multiplier=1.0, policy="protocol", margin=1.0,
                              wallet_mode="prefix_hybrid", safety_factor=1.0),
    "prefix_two_times": dict(wallet_multiplier=1.0, policy="protocol", margin=1.0,
                              wallet_mode="prefix_hybrid", safety_factor=2.0),
    "static_warning_105": dict(wallet_multiplier=1.0, policy="static_warning", margin=1.05),
    "static_warning_110": dict(wallet_multiplier=1.0, policy="static_warning", margin=1.10),
    "dynamic_warning_110": dict(wallet_multiplier=1.0, policy="dynamic_warning", margin=1.10),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def prefix_wallet(profile, cutoff, factor, inferencer_class):
    """Infer both arms' initial funding from the same historical prefix only."""
    import pandas as pd
    transactions = [tx for tx in profile['transactions'] if float(tx['timestamp']) <= cutoff]
    history = pd.DataFrame(transactions).rename(columns={'action': 'type', 'symbol': 'reserve'})
    inferencer = inferencer_class(factor)
    wallet = (inferencer.normalize_wallet_amounts(inferencer.infer_wallet_hybrid(history, history))
              if not history.empty else {})
    return wallet, dict(inferencer.provenance(), cutoff=cutoff, prefix_transactions=len(transactions))


def choose_indices(parent, positives, negatives, horizon_seconds):
    chosen = {True: [], False: []}
    for index in range(12000):
        source = parent / "cases" / f"{index:05d}-core.json"
        if not source.is_file():
            continue
        item = json.loads(source.read_text())
        if not item.get("result", {}).get("success"):
            continue
        arm = item.get("arms", {}).get("without")
        if not arm:
            continue
        category = bool(arm["liquidation_stats"]["liquidated"])
        if category:
            elapsed = arm["liquidation_stats"].get("time_to_liquidation")
            if elapsed is None or not (600 < float(elapsed) <= horizon_seconds + 600):
                continue
        limit = positives if category else negatives
        if len(chosen[category]) < limit:
            chosen[category].append(index)
        if len(chosen[True]) == positives and len(chosen[False]) == negatives:
            break
    return chosen[True] + chosen[False]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=int, default=2592000)
    parser.add_argument("--positives", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--reuse-cache", type=Path, action="append", default=[])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=("reference", "wallet_one_times", "wallet_two_times",
                                 "static_warning_105", "static_warning_110"))
    args = parser.parse_args()
    if args.horizon_seconds <= 0:
        raise ValueError("Positive fixed follow-up required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    indices = args.indices if args.indices is not None else choose_indices(
        args.parent, args.positives, args.negatives, args.horizon_seconds)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("No unique cases selected")
    identity = dict(recommendations=str(args.recommendations.resolve()),
                    recommendations_sha256=sha(args.recommendations),
                    parent=str(args.parent.resolve()),
                    parent_manifest_sha256=sha(args.parent / "manifest.json"),
                    code_sha256=sha(__file__), indices=indices,
                    scenarios=list(args.scenarios), horizon_seconds=args.horizon_seconds,
                    reuse_cache=[str(path.resolve()) for path in args.reuse_cache],
                    wallet_inference_sha256=sha(Path(__file__).resolve().parents[1] / 'Aave-Simulator/profile_gen/wallet_inference.py'),
                    scope="Deterministic first saved parent-control positives within fixed horizon and over 600 seconds, plus first negatives; diagnostic subset, not representative")
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Changed inputs or configuration; use a new output")
    atomic_json(manifest, identity)
    os.environ.update(AAVE_EVALUATION_CACHE=str(output / "cache"),
                      AAVE_SIMULATION_CACHE=str(output / "simulation-cache"),
                      AAVE_RECOMMENDATIONS_FILE=str(args.recommendations.resolve()))
    for key in ("AAVE_SURVIVAL_DATA", "AAVE_CORE_TRANSACTIONS",
                "AAVE_HISTORICAL_EVIDENCE", "AAVE_INDEXED_COVERAGE",
                "AAVE_INDEXED_ASSETS"):
        os.environ.pop(key, None)
    import performSimulations as evaluator
    import utils.simulations as simulations
    from .followup import fixed_followup

    evaluator.outputFile = str(output / "simulation.log")
    evaluator.get_limited_user_profile = fixed_followup(
        evaluator.get_limited_user_profile, args.horizon_seconds)
    items = list(evaluator.recommendations.items())
    if max(indices) >= len(items):
        raise ValueError("Case index outside recommendation file")
    started = time.time()
    completed_this_run = 0
    total = len(indices) * len(args.scenarios)
    for index in indices:
        case_id, pair = items[index]
        for scenario in args.scenarios:
            settings = SCENARIOS[scenario]
            folder = output / "cases" / f"{index:05d}" / scenario
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "result.json"
            if result_path.is_file():
                continue
            arms = {}
            applied = {}
            funding = {}
            original = evaluator.get_simulation_outcome
            atomic_json(output / 'active.json', dict(index=index, scenario=scenario, started_epoch=time.time(), pid=os.getpid()))

            def simulate(recommendation, suffix, **kwargs):
                profile = copy.deepcopy(kwargs["profile"])
                # The evaluator's legacy lookahead differs between arms. Use
                # one absolute follow-up end for every arm and scenario.
                base = max(float(tx["timestamp"]) for tx in profile["transactions"])
                end = float(recommendation["timestamp"]) + args.horizon_seconds
                if base > end:
                    raise ValueError("Profile contains an action after fixed end")
                kwargs["lookahead_seconds"] = max(1, int(end - base))
                profile["initial_wallet"] = {
                    asset: float(amount) * settings["wallet_multiplier"]
                    for asset, amount in profile.get("initial_wallet", {}).items()
                }
                if settings.get("wallet_mode") == "prefix_hybrid":
                    from profile_gen.wallet_inference import WalletInferencer
                    from utils.constants import DEFAULT_TIME_DELTA_SECONDS
                    wallet, provenance = prefix_wallet(profile,
                        float(recommendation['timestamp']) - DEFAULT_TIME_DELTA_SECONDS,
                        settings.get('safety_factor', 1.5), WalletInferencer)
                    if funding and wallet != funding['wallet']:
                        raise ValueError('The two arms inferred different prefix wallets')
                    funding.update(wallet=wallet, provenance=provenance)
                    profile['initial_wallet'] = wallet
                    profile['wallet_inference'] = provenance
                kwargs["profile"] = profile
                kwargs.update(liquidation_policy=settings["policy"],
                              margin_threshold=settings["margin"],
                              use_enhanced_hf=False, oracle_delay_seconds=0,
                              volatility_discount=0)
                if suffix == "with":
                    applied.update(dict(recommendation))
                # Timestamp-by-timestamp simulator stdout can be enormous;
                # preserve structured results and stderr instead.
                with open(os.devnull, "w") as discard, contextlib.redirect_stdout(discard):
                    from .run_existing_baselines import key
                    rel = key(recommendation, suffix, kwargs, simulations.simulation_code_identity())
                    cached = next((root / rel for root in args.reuse_cache if (root / rel).is_file()), None)
                    if cached is not None:
                        with cached.open('rb') as source:
                            value = pickle.load(source)
                    else:
                        value = simulations.get_simulation_outcome(recommendation, suffix, **kwargs)
                arms[suffix] = value
                return value

            evaluator.get_simulation_outcome = simulate
            try:
                result = evaluator.process_recommendation(copy.deepcopy(pair))
                summary = dict(index=index, case_id=str(case_id), scenario=scenario,
                               result_success=bool(result.get("success")),
                               result_error=result.get("error"),
                               result_skipped=bool(result.get("skipped")),
                               funding=funding,
                               applied_recommendation={key: applied.get(key) for key in
                                                       ("user", "timestamp", "Index Event", "reserve",
                                                        "symbol", "amount", "amountUSD", "abstained")},
                               arms={})
                for side, arm in arms.items():
                    stats = arm["liquidation_stats"]
                    summary["arms"][side] = dict(
                        flagged=bool(stats["liquidated"]),
                        evaluation_target=stats.get("evaluation_target"),
                        policy=stats.get("policy"),
                        time_to_flag=stats.get("time_to_liquidation"),
                        checkpoint_timestamp=arm["checkpoint_state"]["timestamp"],
                        checkpoint_debt_usd=arm["checkpoint_state"].get("total_debt_usd"),
                        checkpoint_state=arm['checkpoint_state'],
                        execution_failures=len(arm.get("execution_failures", [])))
            except Exception as error:
                summary = dict(index=index, case_id=str(case_id), scenario=scenario,
                               error=str(error))
            finally:
                evaluator.get_simulation_outcome = original
            atomic_json(result_path, summary)
            completed_this_run += 1
            saved = len(list((output / 'cases').glob('*/*/result.json')))
            atomic_json(output / 'progress.json', dict(total=total, saved=saved,
                completed_this_run=completed_this_run, remaining_this_run=total-saved,
                seconds=time.time()-started, last_index=index, last_scenario=scenario))
            (output / 'active.json').unlink(missing_ok=True)
            print(index, scenario, "complete" if summary.get("result_success") else "not_processed", flush=True)
    rows = [json.loads(path.read_text()) for path in
            sorted((output / "cases").glob("*/*/result.json"))]
    counts = {scenario: dict(cases=sum(r["scenario"] == scenario for r in rows),
                             errors=sum(r["scenario"] == scenario and
                                        not r.get("result_success", False) for r in rows))
              for scenario in args.scenarios}
    atomic_json(output / "summary.json", dict(counts=counts, selected_cases=len(indices),
                                               code_identity=simulations.simulation_code_identity(),
                                               interpretation="Matched scenario flags on a bounded selected subset; warnings are not protocol eligibility or observed executions"))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
