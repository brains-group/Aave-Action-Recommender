"""Compare gross proposed intervention amounts on the common policy cases.

Proposals are not proof of execution, gas expenditure, or net financial value.
The historical agent recommendations may have been resized by the evaluator;
this audit reports the saved proposal as generated.
"""

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from .audit_existing_policies import load_generation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--original-baselines", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.recommendations.open("rb") as source:
        items = list(pickle.load(source).items())
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    missing_state = 0
    agent_checks = collections.Counter()
    records = args.output / "cases.jsonl"
    with records.open("w") as destination:
        for line in (args.common / "common_cases.jsonl").open():
            case = json.loads(line)
            index = case["index"]
            key, (rec, _info) = items[index]
            if str(key) != str(index):
                raise ValueError("Saved recommendation order does not match case index")
            amount = float(rec.get("amountUSD", 0) or 0)
            if amount < 0:
                raise ValueError("Negative proposed agent capital")
            agent_attempted = not bool(rec.get("abstained", False)) and amount > 0
            pack = load_generation(case["case_id"], args.original_baselines, args.recovery)
            if pack is None:
                missing_state += 1
                funds = None
                same_asset_wallet = None
            else:
                funds = sum(max(0, float(asset.get("wallet", 0))) *
                            float(asset.get("price_usd", 0))
                            for asset in pack["state"].get("assets", {}).values())
                symbol = rec.get("symbol") or rec.get("reserve")
                asset = pack["state"].get("assets", {}).get(symbol)
                same_asset_wallet = (max(0, float(asset["wallet"]))
                                     if asset is not None else None)
            requested_units = float(rec.get("amount", 0) or 0)
            above_same_asset = (requested_units > same_asset_wallet +
                                1e-8 * max(1, same_asset_wallet)
                                if same_asset_wallet is not None else None)
            row = dict(case_id=case["case_id"], variant=case["variant"],
                       control=case["control"],
                       agent_rescue=case["control"] and not case["agent_intervention"],
                       agent_attempted=agent_attempted,
                       agent_proposed_capital_usd=amount if agent_attempted else 0,
                       checkpoint_wallet_value_usd=funds,
                       agent_above_checkpoint_wallet=(amount > funds + 1e-8
                                                      if funds is not None else None),
                       agent_above_same_asset_wallet=above_same_asset,
                       agent_requested_units=requested_units,
                       same_asset_checkpoint_wallet_units=same_asset_wallet,
                       policies={name: dict(attempted=proposal["attempted"],
                                            proposed_capital_usd=proposal["proposed_capital_usd"])
                                 for name, proposal in case["policies"].items()})
            destination.write(json.dumps(row) + "\n")
            agent_checks["above_all_wallet_value"] += bool(row["agent_above_checkpoint_wallet"])
            agent_checks["above_same_asset_wallet"] += bool(above_same_asset)
            agent_checks["missing_same_asset_state"] += same_asset_wallet is None
            agent_checks["agent_rescues_over_one_million_usd_proposed"] += (
                row["agent_rescue"] and agent_attempted and amount > 1_000_000)
            for policy, attempted, capital in [
                ("agent", agent_attempted, amount if agent_attempted else 0),
                *((name, proposal["attempted"], proposal["proposed_capital_usd"])
                  for name, proposal in case["policies"].items())]:
                group = groups[policy]
                group["capital_all"].append(capital)
                if attempted:
                    group["capital_attempted"].append(capital)
                if row["control"]:
                    group["capital_control_liquidated"].append(capital)
    summary = {}
    for policy, group in groups.items():
        all_amounts = group["capital_all"]
        attempts = group["capital_attempted"]
        ordered = sorted(attempts)
        summary[policy] = dict(cases=len(all_amounts), attempts=len(attempts),
                               total_proposed_usd=sum(all_amounts),
                               mean_per_case_usd=sum(all_amounts) / len(all_amounts) if all_amounts else None,
                               mean_per_attempt_usd=sum(attempts) / len(attempts) if attempts else None,
                               median_per_attempt_usd=float(np.median(attempts)) if attempts else None,
                               p90_per_attempt_usd=float(np.quantile(attempts, .9)) if attempts else None,
                               p99_per_attempt_usd=float(np.quantile(attempts, .99)) if attempts else None,
                               maximum_proposed_usd=ordered[-1] if ordered else None,
                               share_of_total_in_largest_one_percent=(
                                   sum(ordered[-max(1, len(ordered) // 100):]) / sum(ordered)
                                   if ordered and sum(ordered) else None),
                               attempts_above_one_million_usd=sum(value > 1_000_000 for value in attempts),
                               proposed_on_control_liquidation_usd=sum(group["capital_control_liquidated"]))
    report = dict(groups=summary, missing_checkpoint_wallet_state=missing_state,
                  agent_checkpoint_checks=agent_checks,
                  recommendations_sha256=sha(args.recommendations),
                  common_cases_sha256=sha(args.common / "common_cases.jsonl"),
                  scope="Gross saved proposals on identical case IDs; not actual execution costs or equal-capital policy comparison")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: {name: value[name] for name in
                              ("cases", "attempts", "total_proposed_usd")}
                      for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
