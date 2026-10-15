"""Interpret bounded simulator scenarios without conflating warning and eligibility.

Wallet-scaled scenarios are flagged when they change the historical checkpoint or
execution-failure count. Such cases cannot isolate intervention-funding effects.
"""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument('--reference-scenario', default='reference')
    parser.add_argument("--reference-run", type=Path,
                        help="Completed matched reference run to reuse without recomputation")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    external = None
    if args.reference_run:
        external = json.loads((args.reference_run / "manifest.json").read_text())
        for field in ("recommendations_sha256", "parent_manifest_sha256",
                      "indices", "horizon_seconds"):
            if manifest[field] != external[field]:
                raise ValueError(f"Reference run differs in {field}")
        active_identity = json.loads((args.run / "summary.json").read_text())["code_identity"]
        old_identity = json.loads((args.reference_run / "summary.json").read_text())["code_identity"]
        if active_identity != old_identity:
            raise ValueError("Simulator code identity differs from reference")
    by_case = collections.defaultdict(dict)
    for path in sorted((args.run / "cases").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        by_case[result["index"]][result["scenario"]] = result
    if args.reference_run:
        for index in manifest["indices"]:
            path = args.reference_run / "cases" / f"{index:05d}" / args.reference_scenario / "result.json"
            if path.is_file():
                by_case[index][args.reference_scenario] = json.loads(path.read_text())
    groups = collections.defaultdict(collections.Counter)
    cases = []
    for index in manifest["indices"]:
        reference = by_case[index].get(args.reference_scenario)
        for scenario in manifest["scenarios"]:
            result = by_case[index].get(scenario)
            pair = bool(result and result.get("result_success") and
                        set(result.get("arms", {})) == {"without", "with"})
            ref_pair = bool(reference and reference.get("result_success") and
                            set(reference.get("arms", {})) == {"without", "with"})
            control_changed = None
            if pair and ref_pair:
                current = result["arms"]["without"]
                old = reference["arms"]["without"]
                control_changed = (current["flagged"] != old["flagged"] or
                                   current["checkpoint_debt_usd"] != old["checkpoint_debt_usd"] or
                                   current["execution_failures"] != old["execution_failures"])
            requested_policy = ("protocol" if scenario in
                                {"reference", "wallet_one_times", "wallet_two_times",
                                 "no_inferred_wallet", "prefix_reinferred_wallet",
                                 "prefix_one_times", "prefix_two_times"} else
                                "dynamic_warning" if scenario.startswith("dynamic_") else
                                "static_warning")
            metadata_mismatch = (pair and any(arm.get("policy") != requested_policy
                                              for arm in result["arms"].values()))
            record = dict(index=index, scenario=scenario, paired=pair,
                          error=None if result is None else result.get("result_error") or result.get("error"),
                          requested_policy=requested_policy,
                          reported_policy=None if not pair else result["arms"]["without"].get("policy"),
                          metadata_mismatch=metadata_mismatch,
                          control_changed_from_reference=control_changed,
                          control_flag=None if not pair else result["arms"]["without"]["flagged"],
                          intervention_flag=None if not pair else result["arms"]["with"]["flagged"])
            cases.append(record)
            counts = groups[scenario]
            counts["selected"] += 1
            counts["paired"] += pair
            counts["missing_or_failed"] += not pair
            counts["metadata_mismatch"] += bool(metadata_mismatch)
            counts["control_changed_from_reference"] += bool(control_changed)
            if pair:
                counts["control_flags"] += record["control_flag"]
                counts["intervention_flags"] += record["intervention_flag"]
                counts["flag_removed"] += (record["control_flag"] and
                                           not record["intervention_flag"])
    report = dict(groups=groups, cases=cases, selected_indices=manifest["indices"],
                  external_reference_run=(str(args.reference_run.resolve())
                                          if args.reference_run else None),
                  reference_scenario=args.reference_scenario,
                  interpretation="Protocol and warning flags are different targets. Funding scenarios that change historical checkpoint states or failures measure end-to-end reconstruction sensitivity, not intervention funding alone. Policy metadata mismatches are explicitly counted.")
    (args.run / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
