"""Extract the recent-history window the SHORT-HORIZON models need to serve.

WHY THIS EXISTS.

The 24h and 48h models cannot answer a bare date: they read the counter's own
recent counts as features. Until now the only way to supply those was for the
CALLER to send them in a POST body -- which meant our own frontend fetched
~37 KB of history from the forecast service, extracted 1,320 hours, and POSTed
55 KB of it to the lag service. Our own data, moved between two of our own
machines, through the user's connection.

This script writes that history where the lag service can read it itself, so the
endpoint can be a plain GET with no payload at all.

WHAT IT IS NOT. This is not `data/live` and it does not make the models
generally servable. It is a FROZEN snapshot of the tail of the recorded data,
which lets the lag models answer exactly the dates that sit within one lead of
where the recording stops -- 2026-01-01 and 2026-01-02 today. Answering
2026-03-14 needs an ingestion feed delivering counts on a schedule, which is a
data-supply problem, not a code one. See LEAD_LAG.md, "The spec for data/live".

SHAPE. One file per counter, mirroring actuals/ so the service can read a single
small file per request rather than holding a dictionary of every series in
memory:

    data/history/1410.json
      {"poste_id": 1410,
       "series": {"1-V": {"2025-12-01": [24 ints], ...}},
       "window": {"from": "...", "through": "..."}}

WINDOW SIZE. The 48h model wants 768h (32 days). The default here is wider on
purpose: counters have gaps -- counter 1410 reported 339 of 365 days in 2025 --
and a window cut to exactly 32 calendar days can hold fewer than 768 hours of
actual readings, which the model refuses. The surplus costs kilobytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "actuals"
OUT = ROOT / "data" / "history"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=45,
                    help="calendar days to keep, counting back from --through "
                         "(default 45; the 48h model needs 32)")
    ap.add_argument("--through", default="auto",
                    help="last day to keep, YYYY-MM-DD, or 'auto' for the latest "
                         "day present in actuals/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(SRC.glob("*.json"))
    if not files:
        raise SystemExit(f"no actuals in {SRC}")

    # The window END is global, not per series. A series that stopped reporting
    # in October must NOT get its own later window -- it genuinely cannot answer
    # a January date, and silently giving it a stale window would turn a clean
    # 422 into a forecast built on two-month-old lags.
    through = args.through
    if through == "auto":
        through = ""
        for f in files:
            try:
                d = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            # actuals/ also holds index.json, whose top level is a LIST. Guard
            # here as well as in the main loop below -- this scan runs first, so
            # without it the script dies before reaching that guard.
            if not isinstance(d, dict):
                continue
            for days in d.get("series", {}).values():
                if isinstance(days, dict) and days:
                    through = max(through, max(days))
    if not through:
        raise SystemExit("could not determine the window end")

    import datetime as dt
    end = dt.date.fromisoformat(through)
    start = (end - dt.timedelta(days=args.days - 1)).isoformat()
    print(f"window {start} .. {through}  ({args.days} days)")

    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        for old in OUT.glob("*.json"):
            old.unlink()

    kept_series = kept_values = written = skipped = 0
    for f in files:
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(d, dict) or "poste_id" not in d or "series" not in d:
            skipped += 1                       # e.g. actuals/index.json
            continue
        out: dict[str, dict] = {}
        for key, days in d["series"].items():
            if not isinstance(days, dict):
                continue
            win = {k: v for k, v in days.items() if start <= k <= through}
            if win:
                out[key] = win
                kept_series += 1
                kept_values += sum(len(v) for v in win.values())
        if not out:
            continue                            # stopped reporting before the window
        payload = {"poste_id": d["poste_id"], "series": out,
                   "window": {"from": start, "through": through}}
        if not args.dry_run:
            (OUT / f"{d['poste_id']}.json").write_text(
                json.dumps(payload, separators=(",", ":")))
        written += 1

    size = sum(p.stat().st_size for p in OUT.glob("*.json")) if not args.dry_run else 0
    print(f"  {written} counter files, {kept_series} series, {kept_values:,} hourly values")
    if skipped:
        print(f"  {skipped} source file(s) skipped (not counter actuals)")
    if not args.dry_run:
        print(f"  {size/1024/1024:.2f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
