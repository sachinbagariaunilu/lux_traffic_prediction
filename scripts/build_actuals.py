#!/usr/bin/env python3
"""Turn the raw traffic CSVs into small per-counter JSON files the API can serve.

The raw files are ~150 MB each -- far too big to ship to a browser that only
ever needs 24 numbers at a time. This splits them by POSTE_ID, so a client
fetches one small file instead of the whole year.

Three things happen here, and the splitting is only the last of them:

  1. Rows are dropped. The CSVs carry a 'U' vehicle code the model never saw,
     and series the bundle does not know (the raw data holds ~1,594 V/C series
     across 314 counters; the model knows 1,058 across 270). A recorded count
     with no forecast to compare against has nothing to do.
  2. Columns are dropped. Only the 24 hourly counts survive as data. LOCALITE,
     ROUTE, SENS, COORD_X and COORD_Y are exactly what /counters already
     returns, so keeping them here would duplicate the API; SUM_TRAF is just
     the sum of the 24.
  3. What is left is split one file per counter.

Together that turns ~295 MB of CSV into roughly 30 MB on disk.

Usage -- 2025 only, which is what the frontend offers:

    .venv/bin/python scripts/build_actuals.py \
        --model models/forecast_model_2024.pkl \
        --csv data/raw/donneestrafic-2025-Data.csv \
        --out actuals

Each CSV holds exactly one year, so the year selection *is* the --csv list.
2024 is deliberately left out: it is the training year, so the model reproduces
it rather than forecasting it, and scoring against it flatters the model. Pass
the 2024 CSV as a second --csv to put it back.

Output shape, one file per counter (actuals/1410.json):

    {"poste_id": 1410,
     "series": {"1-V": {"2025-02-15": [24 integers], ...}, ...}}

Served by GET /actuals/{poste_id}. Requires pandas + joblib -- it reads the
pickle. The running API does not; it only reads the JSON.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

HOUR_COLS = [f"P{h:02d}_{h + 1:02d}" for h in range(24)]
KEY_COLS = ["POSTE_ID", "DIRECTION", "VEHICULE"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="forecast_model_2024.pkl")
    ap.add_argument("--csv", action="append", required=True, help="repeatable")
    ap.add_argument("--out", default="actuals")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    known = set(map(tuple, bundle["meta"][KEY_COLS].values))
    print(f"model knows {len(known)} series")

    # poste_id -> "direction-vehicule" -> "YYYY-MM-DD" -> [24 ints]
    store: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for path in a.csv:
        df = pd.read_csv(path, usecols=KEY_COLS + ["DATECOM"] + HOUR_COLS)
        keep = [tuple(r) in known for r in df[KEY_COLS].values]
        df = df[keep].copy()
        df["DATECOM"] = pd.to_datetime(df["DATECOM"], format="%m/%d/%Y").dt.strftime(
            "%Y-%m-%d"
        )
        # Counts are whole vehicles; NaN gaps become 0 so the array stays fixed-width.
        hours = df[HOUR_COLS].fillna(0).round().astype(int).values

        for (pid, direction, veh, day), row in zip(
            df[KEY_COLS + ["DATECOM"]].itertuples(index=False, name=None), hours
        ):
            store[int(pid)][f"{int(direction)}-{veh}"][day] = row.tolist()
        print(f"  {Path(path).name}: {len(df):,} rows")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index = []
    for pid, series in store.items():
        # separators= keeps the files tight; they are machine-read, not browsed.
        (out / f"{pid}.json").write_text(
            json.dumps(
                {"poste_id": pid, "series": series}, separators=(",", ":")
            )
        )
        index.append(pid)

    (out / "index.json").write_text(json.dumps(sorted(index), separators=(",", ":")))

    total = sum(f.stat().st_size for f in out.glob("*.json"))
    print(
        f"wrote {len(index)} counter files + index to {out}  "
        f"({total / 1e6:.1f} MB total, {total / len(index) / 1e3:.0f} KB avg)"
    )


if __name__ == "__main__":
    main()
