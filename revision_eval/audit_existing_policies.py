"""Read-only same-case audit of the existing policy and agent evaluation.

This script never treats an unavailable/error case as safe. It keeps original
and recovered artifacts in place and writes a new, compact case ledger.
Capital is gross proposed/executed action value, not gas or net cost.
"""

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path

from .policies import POLICIES


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_policy(case, policy, original, recovery):
    folders = [original / "cases" / case]
    folders.extend(recovery / name / "cases" / case for name in
                   ("baseline-errors-fixed", "baseline-errors",
                    "baseline-errors-indexed"))
    for folder in folders:
        path = folder / (policy + ".pkl")
        if not path.is_file():
            continue
        try:
            with path.open("rb") as f:
                result = pickle.load(f)
            if (result.get("result", {}).get("success") and
                    set(result.get("arms", {})) == {"without", "with"}):
                return result, str(path)
        except Exception:
            continue
    return None, None


def load_generation(case, original, recovery):
    folders = [original / "cases" / case]
    folders.extend(recovery / name / "cases" / case for name in
                   ("baseline-errors-fixed", "baseline-errors",
                    "baseline-errors-indexed"))
    for folder in folders:
        path = folder / "recommendations.pkl"
        if path.is_file():
            with path.open("rb") as source:
                return pickle.load(source)
    return None


def event_time(arm):
    stats = arm["liquidation_stats"]
    elapsed = stats.get("time_to_liquidation")
    if elapsed is None:
        return None
    return float(arm["checkpoint_state"]["timestamp"]) + float(elapsed)


def compact(record, proposal):
    without, with_action = (record["arms"][side] for side in ("without", "with"))
    rec = record["recommendation"][0]
    action_time = float(rec["timestamp"])
    attempted = proposal.get("action") is not None
    failures = [f for f in with_action.get("execution_failures", [])
                if abs(float(f.get("timestamp", -1)) - action_time) < 1e-6]
    applied = record.get("applied_recommendation", {})
    return dict(control=bool(without["liquidation_stats"]["liquidated"]),
                intervention=bool(with_action["liquidation_stats"]["liquidated"]),
                control_time=event_time(without), intervention_time=event_time(with_action),
                attempted=attempted, execution_failed=bool(failures),
                proposed_capital_usd=float(proposal.get("capital_usd", 0) or 0),
                executed_capital_usd=(float(applied.get("amountUSD", 0) or 0)
                                      if attempted and not failures else 0),
                action=proposal.get("action"),
                action_timestamp=action_time)


def summarize(rows, policy):
    eligible = [r for r in rows if r.get("parent_hf_processed")]
    matched = [r for r in eligible if r.get("parent") is not None and
               policy in r["policies"] and
               r["policies"][policy]["control"] == r["parent"]["control"]]
    control = [r for r in matched if r["parent"]["control"]]
    rescued_agent = sum(not r["parent"]["intervention"] for r in control)
    rescued_policy = sum(not r["policies"][policy]["intervention"] for r in control)
    delay = [r["policies"][policy]["intervention_time"] -
             r["policies"][policy]["control_time"] for r in control
             if r["policies"][policy]["control_time"] is not None and
             r["policies"][policy]["intervention_time"] is not None]
    return dict(parent_hf_processed=len(eligible), matched=len(matched),
                missing_or_control_mismatch=len(eligible) - len(matched),
                matched_control_liquidations=len(control),
                agent_rescues=rescued_agent, policy_rescues=rescued_policy,
                agent_only_rescues=sum(not r["parent"]["intervention"] and
                                       r["policies"][policy]["intervention"] for r in control),
                policy_only_rescues=sum(r["parent"]["intervention"] and
                                        not r["policies"][policy]["intervention"] for r in control),
                policy_worsened=sum(not r["parent"]["control"] and
                                    r["policies"][policy]["intervention"] for r in matched),
                attempted=sum(r["policies"][policy]["attempted"] for r in matched),
                failed_action=sum(r["policies"][policy]["execution_failed"] for r in matched),
                proposed_capital_usd=sum(r["policies"][policy]["proposed_capital_usd"] for r in matched),
                executed_capital_usd=sum(r["policies"][policy]["executed_capital_usd"] for r in matched),
                delayed_at_least_7d=sum(x >= 604800 for x in delay),
                both_liquidated_with_comparable_times=len(delay),
                fee_scenarios_usd={str(fee): fee * sum(r["policies"][policy]["attempted"]
                                                       for r in matched) for fee in (0, 0.1, 1, 5)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("core", "indexed"), required=True)
    parser.add_argument("--limit", type=int, default=12000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(parent=str(args.parent.resolve()),
                    parent_manifest_sha256=sha(args.parent / "manifest.json"),
                    original=str(args.original.resolve()),
                    original_manifest_sha256=sha(args.original / "manifest.json"),
                    recovery=str(args.recovery.resolve()),
                    recovery_report_sha256=sha(args.recovery / "FINAL_RECOVERY_REPORT.json"),
                    variant=args.variant, limit=args.limit,
                    code_sha256=sha(__file__))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Inputs/code changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    ledger = args.output / "cases.jsonl"
    done = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row["index"] in done:
                raise ValueError("Duplicate saved case")
            done[row["index"]] = row
    with ledger.open("a") as target:
        for index in range(args.limit):
            if index in done:
                continue
            case = f"{index:05d}-{args.variant}"
            row = dict(index=index, case_id=case, variant=args.variant,
                       parent_hf_processed=False, parent=None, policies={},
                       sources={})
            parent_path = args.parent / "cases" / (case + ".json")
            if parent_path.is_file():
                parent = json.loads(parent_path.read_text())
                result = parent.get("result", {})
                row["parent_hf_processed"] = bool(result.get("stats_updates_hf_only", {})
                                                  .get("overall", {}).get("processed"))
                if result.get("success") and set(parent.get("arms", {})) == {"without", "with"}:
                    without, with_action = (parent["arms"][side] for side in ("without", "with"))
                    row["parent"] = dict(control=bool(without["liquidation_stats"]["liquidated"]),
                                         intervention=bool(with_action["liquidation_stats"]["liquidated"]),
                                         control_time=event_time(without),
                                         intervention_time=event_time(with_action))
            generation = load_generation(case, args.original, args.recovery)
            if generation is not None:
                proposals = generation.get("proposals", {})
                state = generation.get("state", {})
                row["debt_usd"] = state.get("total_debt_usd")
                row["checkpoint"] = state.get("timestamp")
                for policy in POLICIES:
                    record, path = load_policy(case, policy, args.original, args.recovery)
                    if record is None:
                        continue
                    row["policies"][policy] = compact(record, proposals.get(policy, {}))
                    row["sources"][policy] = path
            target.write(json.dumps(row, default=str) + "\n")
            target.flush()
            done[index] = row
            if (index + 1) % 500 == 0:
                print(f"{index + 1}/{args.limit} cases saved", flush=True)
    rows = [done[index] for index in range(args.limit)]
    report = dict(variant=args.variant, selected=len(rows),
                  parent_hf_processed=sum(r["parent_hf_processed"] for r in rows),
                  policies={policy: summarize(rows, policy) for policy in POLICIES},
                  note="Same-case historical evaluator with original follow-up. Capital is gross action value; fee scenarios are illustrative, not measured gas. Core and indexed remain separate.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(variant=args.variant, selected=len(rows),
                          parent_hf_processed=report["parent_hf_processed"],
                          matched={policy: value["matched"] for policy, value in report["policies"].items()}),
                     indent=2))


if __name__ == "__main__":
    main()
