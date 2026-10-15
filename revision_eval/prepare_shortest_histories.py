"""Extract feature history for a frozen shortest-liquidation cohort.

The source CSVs are read once each. Completed parts are resumable, and the
original feature tables are never modified.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd


EVENTS = ("Borrow", "Deposit", "Repay", "Withdraw", "Liquidated")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(a.cohort, usecols=["user", "timestamp"])
    cutoffs = rows.groupby("user").timestamp.max().to_dict()
    identity = dict(
        cohort_sha256=digest(a.cohort),
        sources={
            f"{ie}/{oe}": (
                lambda st: dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
            )((a.data / ie / oe / "data.csv").stat())
            for ie in EVENTS for oe in EVENTS
        },
        code_sha256=digest(Path(__file__)),
    )
    manifest = a.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("History inputs changed; choose a fresh output directory")
    else:
        manifest.write_text(json.dumps(identity, indent=2))
    for ie in EVENTS:
        for oe in EVENTS:
            source = a.data / ie / oe / "data.csv"
            dest = a.output / f"{ie}-{oe}.pkl"
            if dest.exists():
                print("cached", ie, oe, flush=True)
                continue
            parts = []
            for chunk in pd.read_csv(source, chunksize=100_000):
                chunk = chunk.loc[chunk.user.isin(cutoffs)]
                if not chunk.empty:
                    chunk = chunk.loc[chunk.timestamp <= chunk.user.map(cutoffs)]
                    if not chunk.empty:
                        parts.append(chunk)
            selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            tmp = dest.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(selected, f, protocol=4)
            tmp.replace(dest)
            print(ie, oe, len(selected), flush=True)
    (a.output / "complete.json").write_text(json.dumps({"parts": len(EVENTS)**2}))


if __name__ == "__main__":
    main()
