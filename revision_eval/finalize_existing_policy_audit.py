"""One common-case comparison for agent and all four intervention baselines.

The parent no-action arm supplies the control. This does not infer missing
policy outcomes from unavailable cases. Submitted, nonfailed action value is
reported as an estimate, since the saved applied-action ledger lacks per-action
timestamp/amount confirmation for every recommendation.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path


POLICIES = ("static_hf", "repay_only", "deposit_only", "automation_style")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.ledger / "summary.json").is_file():
        raise ValueError("Source audit incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = eligible = with_parent = common = 0
    counts = {policy: collections.Counter() for policy in ("agent", *POLICIES)}
    records = args.output / "common_cases.jsonl"
    with records.open("w") as target:
        for line in (args.ledger / "cases.jsonl").open():
            row = json.loads(line)
            selected += 1
            if not row.get("parent_hf_processed"):
                continue
            eligible += 1
            parent = row.get("parent")
            if parent is None:
                continue
            with_parent += 1
            if any(policy not in row["policies"] or
                   row["policies"][policy]["control"] != parent["control"]
                   for policy in POLICIES):
                continue
            common += 1
            result = dict(case_id=row["case_id"], index=row["index"],
                          variant=row["variant"], control=parent["control"],
                          agent_intervention=parent["intervention"],
                          policies={})
            for policy in ("agent", *POLICIES):
                if policy == "agent":
                    intervention = parent["intervention"]
                    proposal = None
                else:
                    proposal = row["policies"][policy]
                    intervention = proposal["intervention"]
                summary = counts[policy]
                summary["cases"] += 1
                summary["control_liquidations"] += parent["control"]
                summary["rescued"] += parent["control"] and not intervention
                summary["worsened"] += not parent["control"] and intervention
                if proposal is not None:
                    summary["attempted"] += proposal["attempted"]
                    summary["failed_action"] += proposal["execution_failed"]
                    summary["proposed_capital_usd"] += proposal["proposed_capital_usd"]
                    summary["submitted_nonfailed_capital_usd_estimate"] += proposal["executed_capital_usd"]
                    summary["agent_only_rescues"] += (parent["control"] and
                                                      not parent["intervention"] and
                                                      intervention)
                    summary["policy_only_rescues"] += (parent["control"] and
                                                       parent["intervention"] and
                                                       not intervention)
                    result["policies"][policy] = dict(intervention=intervention,
                                                       attempted=proposal["attempted"],
                                                       execution_failed=proposal["execution_failed"],
                                                       proposed_capital_usd=proposal["proposed_capital_usd"])
            target.write(json.dumps(result) + "\n")
    result = dict(selected=selected, parent_hf_processed=eligible,
                  parent_paired=with_parent, common_agent_and_four_policies=common,
                  excluded_from_common=eligible-common,
                  policies=counts,
                  source=str(args.ledger.resolve()),
                  source_ledger_sha256=sha(args.ledger / "cases.jsonl"),
                  interpretation="All policy rows use the same case IDs and agree with the parent control liquidation flag. Complete checkpoint-state equality has not been proved. Outcome is simulated eligibility under the historical evaluator and variable follow-up; no on-chain executions are inferred. Capital is gross proposed amount, not gas or net cost.")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(selected=selected, parent_hf_processed=eligible,
                          common_agent_and_four_policies=common,
                          control_liquidations=counts["agent"]["control_liquidations"],
                          rescues={policy: data["rescued"] for policy, data in counts.items()}),
                     indent=2))


if __name__ == "__main__":
    main()
