"""Audit saved inferred wallets on largest existing-case agent proposals.

This is a read-only reproducibility check. The current inference code may differ
from the code used to build old profiles; differences do not establish causation.
"""

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

from .audit_existing_policies import load_generation


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def profile_path(root, account):
    matches = [root / group / "profiles" / f"user_{account}.json"
               for group in ("liquidated_profiles", "non_liquidated_profiles")]
    return [path for path in matches if path.is_file()]


def inferred_from_actions(transactions, inferencer):
    frame = pd.DataFrame(transactions)
    if frame.empty:
        return {}
    frame = frame.rename(columns={"action": "type", "symbol": "reserve"})
    return inferencer.infer_wallet_hybrid(frame, frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital-cases", type=Path, required=True)
    parser.add_argument("--original-baselines", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--selection", choices=("largest", "random"), default="largest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("--top must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Aave-Simulator"))
    from profile_gen.wallet_inference import WalletInferencer

    cases = [json.loads(line) for line in args.capital_cases.open()
             if line.strip()]
    cases = [row for row in cases if row["agent_attempted"]]
    if args.selection == "random":
        cases = random.Random(args.seed).sample(cases, min(args.top, len(cases)))
    else:
        cases = sorted(cases,
                       key=lambda row: (-row["agent_proposed_capital_usd"],
                                        row["case_id"]))[:args.top]
    inferencer = WalletInferencer(1.5)
    counts = collections.Counter()
    output = args.output / "cases.jsonl"
    with output.open("w") as destination:
        for item in cases:
            record = dict(case_id=item["case_id"],
                          proposed_usd=item["agent_proposed_capital_usd"])
            try:
                pack = load_generation(item["case_id"], args.original_baselines,
                                       args.recovery)
                if pack is None:
                    raise ValueError("generation artifact unavailable")
                account = pack["recommendations"]["control"][0]["user"].lower()
                candidates = profile_path(args.profiles, account)
                if not candidates:
                    raise ValueError("profile missing")
                # Match utils.data.get_user_profile: non-liquidated first.
                path = candidates[0]
                profile = json.loads(path.read_text())
                actions = profile.get("transactions", [])
                checkpoint = int(pack["state"]["timestamp"])
                prefix = [tx for tx in actions if int(tx["timestamp"]) <= checkpoint]
                current_full = inferred_from_actions(actions, inferencer)
                current_prefix = inferred_from_actions(prefix, inferencer)
                saved = profile.get("initial_wallet", {})
                ratios = {asset: saved_amount / max(current_full.get(asset, 0), 1e-12)
                          for asset, saved_amount in saved.items()
                          if saved_amount > 0}
                record.update(account=account, profile=str(path.resolve()),
                              duplicate_profile_paths=[str(p.resolve()) for p in candidates[1:]],
                              profile_sha256=digest(path), checkpoint=checkpoint,
                              actions_total=len(actions), actions_through_checkpoint=len(prefix),
                              future_actions=len(actions)-len(prefix),
                              saved_initial_wallet=saved,
                              inferred_full_current=current_full,
                              inferred_prefix_current=current_prefix,
                              saved_to_current_full_ratios=ratios,
                              any_saved_over_10x_current=any(value > 10 for value in ratios.values()),
                              any_saved_over_100x_current=any(value > 100 for value in ratios.values()),
                              any_full_over_10x_prefix=any(
                                  value > 10 * max(current_prefix.get(asset, 0), 1e-12)
                                  for asset, value in current_full.items()),
                              profile_description=profile.get("description"))
                counts["profile_found"] += 1
                counts["duplicate_profile_paths"] += len(candidates) > 1
                for key in ("any_saved_over_10x_current", "any_saved_over_100x_current",
                            "any_full_over_10x_prefix"):
                    counts[key] += bool(record[key])
                counts["profile_contains_future_actions"] += bool(record["future_actions"])
            except Exception as error:
                record["error"] = str(error)
                counts["error"] += 1
            destination.write(json.dumps(record, allow_nan=False) + "\n")
    result = dict(top_requested=args.top, selection=args.selection, seed=args.seed,
                  selected=len(cases), counts=counts,
                  inputs=dict(capital_cases=str(args.capital_cases.resolve()),
                              capital_cases_sha256=digest(args.capital_cases),
                              code_sha256=digest(__file__)),
                  interpretation="Saved initial wallets versus current inference on saved profile actions. Full-action inference can use actions after the recommendation checkpoint. Profile/code vintages may differ. No saved profile or simulation was changed.")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(selected=len(cases), counts=counts), indent=2))


if __name__ == "__main__":
    main()
