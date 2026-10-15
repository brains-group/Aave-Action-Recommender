"""Join saved-wallet reproducibility diagnostics to paired agent outcomes.

This quantifies exposure to a funding assumption; it does not estimate a
corrected rescue rate or prove that every exposed action used the affected asset.
"""

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_unique(path):
    result = {}
    for line in Path(path).open():
        row = json.loads(line)
        key = row["case_id"]
        if key in result:
            raise ValueError(f"Duplicate case ID: {key}")
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wallet", type=Path, required=True)
    parser.add_argument("--capital", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wallet_path = args.wallet / "cases.jsonl"
    capital_path = args.capital / "cases.jsonl"
    if not (args.wallet / "summary.json").is_file():
        raise ValueError("Wallet audit incomplete")
    if not (args.capital / "summary.json").is_file():
        raise ValueError("Capital audit incomplete")
    wallets, capitals = read_unique(wallet_path), read_unique(capital_path)
    with args.recommendations.open("rb") as source:
        recommendations = list(pickle.load(source).items())
    attempted = {key for key, row in capitals.items() if row["agent_attempted"]}
    if attempted != wallets.keys():
        raise ValueError("Wallet and capital attempted-case sets differ")
    groups = collections.defaultdict(collections.Counter)
    for key in attempted:
        capital, wallet = capitals[key], wallets[key]
        group = groups["rescue" if capital["agent_rescue"] else "other_attempted"]
        group["cases"] += 1
        group["wallet_audit_error"] += "error" in wallet
        if "error" in wallet:
            continue
        index = int(key.split("-", 1)[0])
        rec = recommendations[index][1][0]
        if str(rec.get("user", "")).lower() != wallet["account"]:
            raise ValueError(f"Recommendation/account mismatch for {key}")
        action_symbol = rec.get("symbol") or rec.get("reserve")
        same_asset_ratio = wallet["saved_to_current_full_ratios"].get(action_symbol)
        if same_asset_ratio is None:
            group["proposal_asset_missing_from_saved_wallet"] += 1
        else:
            group["proposal_asset_wallet_over_10x_current"] += same_asset_ratio > 10
            group["proposal_asset_wallet_over_100x_current"] += same_asset_ratio > 100
        group["saved_wallet_over_10x_current"] += wallet["any_saved_over_10x_current"]
        group["saved_wallet_over_100x_current"] += wallet["any_saved_over_100x_current"]
        group["full_inference_over_10x_prefix"] += wallet["any_full_over_10x_prefix"]
        group["profile_has_future_actions"] += bool(wallet["future_actions"])
        group["proposal_above_one_million_usd"] += capital["agent_proposed_capital_usd"] > 1_000_000
        group["proposal_above_checkpoint_wallet_value"] += bool(
            capital["agent_above_checkpoint_wallet"])
        group["proposal_above_same_asset_checkpoint_wallet"] += bool(
            capital["agent_above_same_asset_wallet"])
        group["over_10x_and_above_one_million"] += (
            wallet["any_saved_over_10x_current"] and
            capital["agent_proposed_capital_usd"] > 1_000_000)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(groups=groups,
                  all_common_cases=len(capitals), attempted=len(attempted),
                  source_sha256={str(wallet_path.resolve()): digest(wallet_path),
                                 str(capital_path.resolve()): digest(capital_path),
                                 str(args.recommendations.resolve()): digest(args.recommendations)},
                  interpretation="Reports both any-asset and proposed-action-asset saved-wallet discrepancies. The latter identifies the symbol used by the saved recommendation, but does not prove execution depended on the inflated balance. Saved profiles may use earlier generator code and future actions. These counts identify exposure, not counterfactual outcome changes.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(groups=groups, attempted=len(attempted)), indent=2))


if __name__ == "__main__":
    main()
