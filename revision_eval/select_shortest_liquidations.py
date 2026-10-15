"""Select a separate liquidation stress test by shortest observed event time.

This implements the 300-shortest-per-liquidation-ending-pair part of the
commented original selector, without treating it as code that generated the
existing paper cohort. It never changes the saved ordinary evaluation inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ORIGINS = ("Borrow", "Deposit", "Repay", "Withdraw")


def choose(source: Path, start: float, end: float, count: int, chunksize: int = 100_000):
    selected = pd.DataFrame()
    seen = 0
    eligible = 0
    for chunk in pd.read_csv(source, chunksize=chunksize):
        chunk["source_row"] = range(seen, seen + len(chunk))
        seen += len(chunk)
        keep = chunk.loc[
            (chunk["timestamp"] >= start)
            & (chunk["timestamp"] < end)
            & (chunk["status"] == 1)
            & (chunk["timeDiff"] >= 0)
            & chunk["timeDiff"].notna()
        ]
        eligible += len(keep)
        if keep.empty:
            continue
        selected = pd.concat((selected, keep), ignore_index=True)
        selected = selected.sort_values(
            ["timeDiff", "timestamp", "user", "source_row"], kind="mergesort"
        ).head(count)
    return selected.reset_index(drop=True), {"source_rows": seen, "eligible_events": eligible}


def select(data: Path, output: Path, start: float, end: float, per_pair: int):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not start < end or per_pair <= 0:
        raise ValueError("Expected start < end and per_pair > 0")
    output.mkdir(parents=True)
    batches = []
    sources = []
    for origin in ORIGINS:
        source = data / origin / "Liquidated" / "data.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows, counts = choose(source, start, end, per_pair)
        rows.insert(0, "stress_origin", origin)
        rows.insert(1, "case_id", [f"{origin}:Liquidated:{i}" for i in rows.source_row])
        batches.append(rows)
        st = source.stat()
        sources.append(
            dict(
                path=str(source.resolve()),
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                selected=len(rows),
                **counts,
            )
        )
        print(f"{origin}->Liquidated: selected {len(rows)} / {counts['eligible_events']} eligible", flush=True)
    cohort = pd.concat(batches, ignore_index=True)
    cohort.to_csv(output / "high_risk.csv", index=False)
    manifest = dict(
        method="shortest observed timeDiff per liquidation-ending event pair",
        origins=list(ORIGINS),
        per_pair=per_pair,
        start_inclusive=start,
        end_exclusive=end,
        requirements="status=1 and nonnegative observed timeDiff",
        ties="timestamp, user, original source row",
        uniqueness="source rows; repeated accounts are retained and reported",
        selected=len(cohort),
        distinct_accounts=int(cohort.user.nunique()),
        rapid_within_600_seconds=int((cohort.timeDiff <= 600).sum()),
        sources=sources,
        selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("selected", "distinct_accounts", "rapid_within_600_seconds")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--per-pair", type=int, default=300)
    a = p.parse_args()
    select(a.data, a.output, a.start, a.end, a.per_pair)


if __name__ == "__main__":
    main()
