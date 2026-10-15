"""Matched existing-case policy outcomes by strictly trailing market conditions.

Uses the completed read-only policy ledger. Volatility cut points are estimated
before the earliest original-selection checkpoint; each case sees only quotes
at or before its checkpoint. These are selected-case diagnostics, not a claim
of population representativeness or independent oracle validation.
"""

import argparse
import bisect
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def trailing_volatility(prices, times, timestamp):
    samples = []
    for day in range(7, -1, -1):
        target = timestamp - day * 86400
        index = bisect.bisect_right(times, target) - 1
        if index < 0 or target - times[index] > 86400:
            return None
        samples.append(prices[times[index]])
    if min(samples) <= 0:
        return None
    return float(np.std(np.diff(np.log(samples)), ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-cutoff", type=int, required=True)
    args = parser.parse_args()
    if not (args.ledger / "summary.json").exists():
        raise ValueError("Policy ledger is not finalized")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Aave-Simulator"))
    from simulator.utils import get_price_history

    prices = get_price_history()["WETH"]
    times = sorted(prices)
    first_price = times[0]
    training = [trailing_volatility(prices, times, timestamp)
                for timestamp in range(max(first_price + 8 * 86400,
                                           args.training_cutoff - 365 * 86400),
                                       args.training_cutoff, 86400)]
    training = [value for value in training if value is not None]
    if len(training) < 20:
        raise ValueError("Insufficient pre-selection price coverage")
    low, high = [float(x) for x in np.quantile(training, [1 / 3, 2 / 3])]
    groups = collections.defaultdict(collections.Counter)
    observed = collections.Counter()
    for line in (args.ledger / "cases.jsonl").open():
        row = json.loads(line)
        if not row.get("parent_hf_processed") or not row.get("parent"):
            continue
        timestamp = row.get("checkpoint")
        debt = row.get("debt_usd")
        volatility = trailing_volatility(prices, times, int(timestamp)) if timestamp is not None else None
        regime = ("missing" if volatility is None else "low" if volatility <= low
                  else "middle" if volatility <= high else "high")
        size = ("missing" if debt is None else "<100" if debt < 100 else
                "100-1000" if debt < 1000 else "1000-10000" if debt < 10000
                else ">=10000")
        observed[(regime, size)] += 1
        for policy, result in row["policies"].items():
            if result["control"] != row["parent"]["control"]:
                continue
            for dimension, value in (("volatility", regime), ("debt_usd", size)):
                counts = groups[(policy, dimension, value)]
                counts["matched"] += 1
                if result["control"]:
                    counts["control_liquidations"] += 1
                    counts["agent_rescues"] += not row["parent"]["intervention"]
                    counts["policy_rescues"] += not result["intervention"]
                counts["policy_worsened"] += (not result["control"] and
                                               result["intervention"])
                counts["attempted"] += bool(result["attempted"])
                counts["execution_failed"] += bool(result["execution_failed"])
                counts["proposed_capital_usd"] += result["proposed_capital_usd"]
                counts["executed_capital_usd"] += result["executed_capital_usd"]
    report = dict(source=str(args.ledger.resolve()),
                  ledger_sha256=sha(args.ledger / "cases.jsonl"),
                  training_cutoff=args.training_cutoff, training_days=len(training),
                  volatility_tertiles=[low, high],
                  checkpoint_distribution={f"{regime}:{size}": count
                                           for (regime, size), count in observed.items()},
                  groups={":".join(key): count for key, count in groups.items()},
                  units="gross USD action amount and case-level simulated eligibility; volatility is trailing seven-day WETH daily-log-return SD",
                  limitations="Original selected cohort; profile coverage, pricing, and outcome reconstruction are not independent on-chain truth")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(training_days=len(training), matched_groups=len(groups),
                          volatility_tertiles=[low, high]), indent=2))


if __name__ == "__main__":
    main()
