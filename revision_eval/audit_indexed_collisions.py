"""Classify all unresolved indexed symbol collisions from historical snapshots.

The result does not relax simulator identity guards or treat a zero checkpoint
balance as proof that symbol-only replay is safe. Snapshot cache is read-only.
"""

import argparse
import collections
import hashlib
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_rows(database, account, checkpoint):
    # Stream in source execution order instead of retaining hundreds of thousands
    # of JSON objects for a large account. SQLite may spill sorting to disk.
    query = """SELECT payload FROM rows WHERE entity=? AND id>=? AND id<?
        AND CAST(json_extract(payload, '$.timestamp') AS INTEGER) < ?
        ORDER BY CAST(json_extract(payload, '$.blockNumber') AS INTEGER),
                 CAST(json_extract(payload, '$.logIndex') AS INTEGER), id"""
    for (payload,) in database.execute(query, ("positionSnapshots", account+"-", account+".", checkpoint)):
        yield json.loads(payload)


def positive_underlyings(latest, assets, account, position_parts):
    amounts = collections.defaultdict(Decimal)
    for position, balance in latest.items():
        market, _ = position_parts(position, account)
        asset = assets[market]
        amounts[(asset["symbol"], asset["id"].lower())] += balance
    by_symbol = collections.defaultdict(list)
    for (symbol, underlying), amount in amounts.items():
        if amount > 0:
            by_symbol[symbol].append(underlying)
    return {symbol: sorted(set(ids)) for symbol, ids in by_symbol.items()
            if len(set(ids)) > 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--core-evaluation", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-nonerror-from", type=Path,
                        help="Reuse previously classified rows; retry only errors")
    parser.add_argument("--max-snapshots", type=int, default=100000,
                        help="Keep unusually large account histories unresolved in this diagnostic")
    args = parser.parse_args()
    if args.max_snapshots < 1:
        raise ValueError("--max-snapshots must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Aave-Simulator"))
    from analysis.indexed_history import position_parts
    from analysis.asset_registry import polygon_identity_mappings
    import pickle

    report = json.loads(args.recovery_report.read_text())
    unresolved = [row for row in report["unresolved"]
                  if "Different underlying assets share a symbol" in row["reason"]]
    coverage = json.loads(args.coverage.read_text())
    source = Path(coverage["entities"]["positionSnapshots"]["cache"])
    assets = json.loads(args.assets.read_text())
    assets.update({market: value for market, value in polygon_identity_mappings().items()
                   if market not in assets})
    with args.recommendations.open("rb") as handle:
        recommendations = list(pickle.load(handle).items())
    manifest = dict(recovery_report_sha256=sha(args.recovery_report),
                    recommendations_sha256=sha(args.recommendations),
                    core_manifest_sha256=sha(args.core_evaluation / "manifest.json"),
                    coverage_sha256=sha(args.coverage), assets_sha256=sha(args.assets),
                    snapshot_cache=str(source.resolve()),
                    code_sha256=sha(__file__), selected=len(unresolved),
                    reuse_nonerror_from=(str(args.reuse_nonerror_from.resolve())
                                         if args.reuse_nonerror_from else None),
                    max_snapshots=args.max_snapshots)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Inputs/code changed; select a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    saved = {}
    rows_path = args.output / "cases.jsonl"
    if args.reuse_nonerror_from and not rows_path.exists():
        previous = json.loads((args.reuse_nonerror_from / "manifest.json").read_text())
        for field in ("recovery_report_sha256", "recommendations_sha256",
                      "core_manifest_sha256", "coverage_sha256", "assets_sha256",
                      "snapshot_cache", "selected"):
            if previous[field] != manifest[field]:
                raise ValueError(f"Prior indexed audit differs in {field}")
        previous_rows = [json.loads(line) for line in
                         (args.reuse_nonerror_from / "cases.jsonl").open()]
        if len(previous_rows) != len(unresolved):
            raise ValueError("Prior indexed audit is incomplete")
        with rows_path.open("w") as target:
            for row in previous_rows:
                if "error" not in row:
                    target.write(json.dumps(row) + "\n")
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            record = json.loads(line)
            if record["index"] in saved:
                raise ValueError("Duplicate saved case")
            saved[record["index"]] = record
    uri = source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as database, rows_path.open("a") as target:
        for number, item in enumerate(unresolved, 1):
            index = item["index"]
            if index in saved:
                continue
            rec = recommendations[index][1][0]
            account = str(rec["user"]).lower()
            core_path = args.core_evaluation / "cases" / f"{index:05d}-core.json"
            record = dict(index=index, account=account)
            try:
                core = json.loads(core_path.read_text())
                without = core.get("arms", {}).get("without")
                if without:
                    checkpoint = int(without["checkpoint_state"]["timestamp"])
                    checkpoint_source = "completed_core_control"
                else:
                    # utils.simulations.get_limited_user_profile uses the
                    # decision time minus DEFAULT_TIME_DELTA_SECONDS=600.
                    from utils.constants import DEFAULT_TIME_DELTA_SECONDS
                    checkpoint = int(float(rec["timestamp"]) - DEFAULT_TIME_DELTA_SECONDS)
                    checkpoint_source = "recommendation_minus_evaluator_delta"
                record["checkpoint_source"] = checkpoint_source
                count = database.execute(
                    "SELECT COUNT(*) FROM rows WHERE entity=? AND id>=? AND id<?",
                    ("positionSnapshots", account + "-", account + ".")).fetchone()[0]
                if count > args.max_snapshots:
                    raise ValueError(
                        f"snapshot history {count} exceeds audit cap {args.max_snapshots}")
                history = snapshot_rows(database, account, checkpoint)
                latest = {}
                historical_collision = False
                historical_symbols = set()
                totals = collections.defaultdict(Decimal)
                symbol_positive = collections.defaultdict(set)
                snapshot_count = 0
                for row in history:
                    snapshot_count += 1
                    position = row["position"]["id"]
                    market, _ = position_parts(position, account)
                    if market not in assets:
                        raise ValueError("Missing asset mapping: " + market)
                    asset = assets[market]
                    key = (asset['symbol'], asset['id'].lower())
                    balance = Decimal(str(row['balance']))
                    totals[key] += balance - latest.get(position, Decimal(0))
                    latest[position] = balance
                    if totals[key] > 0:
                        symbol_positive[key[0]].add(key[1])
                    else:
                        symbol_positive[key[0]].discard(key[1])
                    if len(symbol_positive[key[0]]) > 1:
                        historical_collision = True
                        historical_symbols.add(key[0])
                checkpoint_overlaps = positive_underlyings(latest, assets, account, position_parts)
                markets = {position_parts(position, account)[0] for position in latest}
                all_symbols = collections.defaultdict(set)
                for market in markets:
                    asset = assets[market]
                    all_symbols[asset["symbol"]].add(asset["id"].lower())
                record.update(checkpoint=checkpoint, snapshots=snapshot_count,
                              markets=len(markets),
                              configured_colliding_symbols=sorted(symbol for symbol, ids in all_symbols.items()
                                                                  if len(ids) > 1),
                              positive_collision_at_checkpoint=bool(checkpoint_overlaps),
                              positive_collision_ever_before_checkpoint=historical_collision,
                              checkpoint_overlaps=checkpoint_overlaps,
                              historical_collision_symbols=sorted(historical_symbols))
            except Exception as error:
                record["error"] = str(error)
            target.write(json.dumps(record) + "\n")
            target.flush()
            saved[index] = record
            if number % 100 == 0:
                print(f"{number}/{len(unresolved)} classified", flush=True)
    records = [saved[item["index"]] for item in unresolved]
    counts = collections.Counter()
    for row in records:
        if "error" in row:
            counts["error"] += 1
        elif row["positive_collision_at_checkpoint"]:
            counts["positive_at_checkpoint"] += 1
        elif row["positive_collision_ever_before_checkpoint"]:
            counts["positive_only_historical"] += 1
        else:
            counts["no_positive_overlap_in_snapshots"] += 1
    summary = dict(selected=len(unresolved), classified=counts,
                   interpretation="Display-symbol ambiguity remains a data-identity problem in every category; this audit does not authorize symbol aggregation or replaying same-time snapshots.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
