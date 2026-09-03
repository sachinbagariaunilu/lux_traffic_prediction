"""Decide which counters the UI may show, and record why for the rest.

    python scripts/build_counter_manifest.py

THE RULE. The UI compares a 2024-trained forecast against 2025 recorded counts.
A series is only useful there if BOTH sides exist:

    in the model   -> it has a 2024 profile, so it can be forecast
    in actuals     -> it recorded 2025 hours, so the forecast can be SCORED

A series present in only one side is not a bug, it is a counter that was
installed, retired, or reconfigured between the two years. Showing it produces a
panel with half the answer missing, which reads as a broken chart rather than as
a data boundary. So the manifest keeps the INTERSECTION and writes every
exclusion down with its reason.

Writes:
    counter_manifest.json   what the API serves -- the included series
    COUNTER_MANIFEST.md     the human-readable log: counts, reasons, the lists

Re-run whenever the model or actuals/ change. Both outputs are committed, so a
change in what the UI shows is visible in a diff rather than discovered.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The bundle pickles a forecast.features.Profiles dataclass, so `forecast` must
# be IMPORTABLE before joblib.load() -- otherwise unpickling raises
# ModuleNotFoundError. uvicorn puts the app root on sys.path for the server, but
# a script run as scripts/x.py gets scripts/ as sys.path[0], not the root.
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402  -- must follow the sys.path insert
MODEL = ROOT / "models" / "forecast_model.pkl"
ACTUALS = ROOT / "actuals"
OUT_JSON = ROOT / "counter_manifest.json"
OUT_MD = ROOT / "COUNTER_MANIFEST.md"

# A series with only a handful of recorded days can be shown, but almost every
# date the user picks will have no actual line. Flagged, not excluded -- the UI
# already explains a missing day, and dropping a real counter hides coverage.
THIN_DAYS = 30


def main() -> None:
    bundle = joblib.load(MODEL)
    prof = bundle["profiles"]
    model_series = {
        (int(r.POSTE_ID), int(r.DIRECTION), str(r.VEHICULE))
        for r in prof.provenance.itertuples()
    }

    # actuals/<poste>.json -> {"series": {"<dir>-<veh>": {"YYYY-MM-DD": [...]}}}
    # index.json is a plain list of ids, not a counter file -- skip it rather
    # than special-casing the shape, so a future sidecar file cannot break this.
    actual_series: dict[tuple[int, int, str], int] = {}
    for path in sorted(ACTUALS.glob("*.json")):
        if path.stem == "index":
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "poste_id" not in payload:
            print(f"  skipping {path.name}: not a counter file")
            continue
        pid = int(payload["poste_id"])
        for key, days in payload.get("series", {}).items():
            direction, vehicule = key.split("-", 1)
            actual_series[(pid, int(direction), vehicule)] = len(days)

    both = sorted(model_series & set(actual_series))
    model_only = sorted(model_series - set(actual_series))
    actual_only = sorted(set(actual_series) - model_series)

    # Route / locality / coordinates. NOT in the bundle -- it carries only what
    # the model reads, and the model never sees them (see FINDINGS: ROAD_CLASS
    # measured -0.02). They are exported alongside so the API can label a
    # counter without the training repo present.
    attrs = {
        (int(a["poste_id"]), int(a["direction"]), str(a["vehicule"])): a
        for a in json.loads((ROOT / "data" / "external" / "series_attrs.json")
                            .read_text())
    }

    prov = {
        (int(r.POSTE_ID), int(r.DIRECTION), str(r.VEHICULE)): r
        for r in prof.provenance.itertuples()
    }

    def entry(key):
        p, d, v = key
        a = attrs.get(key, {})
        pr = prov.get(key)
        # Mean over the hours the counter actually reported in 2024, not over
        # the calendar year -- no counter reported all 366 days.
        avg = float(prof.series.loc[
            (prof.series.POSTE_ID == p) & (prof.series.DIRECTION == d)
            & (prof.series.VEHICULE == v), "PROF_MEAN"].iloc[0])
        return {
            "poste_id": p, "direction": d, "vehicule": v,
            "route": a.get("route"), "localite": a.get("localite"),
            "sens": a.get("sens"),
            # LUREF (EPSG:2169) METRES, as recorded in the source CSV.
            # NOT lat/lon -- reproject before putting on a map.
            "coord_x": a.get("coord_x"), "coord_y": a.get("coord_y"),
            "avg_per_hour": round(avg, 1),
            # Field names kept as the UI already reads them. days_reported /
            # first_day / last_day describe 2024 (the training year); the 2025
            # figure is additive so nothing downstream breaks.
            "days_reported": int(pr.N_OBS // 24) if pr is not None else None,
            "first_day": str(pr.FIRST_SEEN.date()) if pr is not None else None,
            "last_day": str(pr.LAST_SEEN.date()) if pr is not None else None,
            "recorded_days_2025": actual_series[key],
            "thin": actual_series[key] < THIN_DAYS,
        }

    included = [entry(k) for k in both]
    included.sort(key=lambda e: e["avg_per_hour"], reverse=True)

    OUT_JSON.write_text(json.dumps({
        "rule": "series present in BOTH the 2024 model and 2025 actuals",
        "thin_days_threshold": THIN_DAYS,
        "counts": {"included": len(both), "model_only": len(model_only),
                   "actuals_only": len(actual_only),
                   "model_total": len(model_series),
                   "actuals_total": len(actual_series)},
        "included": included,
        "excluded_model_only": [
            {"poste_id": p, "direction": d, "vehicule": v,
             "reason": "no 2025 recorded data -- forecastable but unscoreable"}
            for p, d, v in model_only],
        "excluded_actuals_only": [
            {"poste_id": p, "direction": d, "vehicule": v,
             "reason": "not in the 2024 training data -- the model refuses it"}
            for p, d, v in actual_only],
    }, indent=2), encoding="utf-8")

    def sites(keys):
        return sorted({p for p, _, _ in keys})

    def group(keys):
        out = defaultdict(list)
        for p, d, v in keys:
            out[p].append(f"{d}-{v}")
        return out

    thin = [i for i in included if i["thin"]]
    md = [
        "# Counter manifest",
        "",
        "Which counters the UI shows, and why the rest are held back.",
        f"Generated by `scripts/build_counter_manifest.py` from",
        f"`models/forecast_model.pkl` and `actuals/`.",
        "",
        "## The rule",
        "",
        "The UI scores a **2024-trained forecast** against **2025 recorded",
        "counts**. Both sides must exist or the panel shows half an answer:",
        "",
        "| Side | Gives us |",
        "| --- | --- |",
        "| 2024 model | a profile, so the series can be forecast |",
        "| 2025 actuals | recorded hours, so the forecast can be scored |",
        "",
        "Only the **intersection** is shown. Everything else is listed below with",
        "its reason — these are counters installed, retired or reconfigured",
        "between the two years, not data errors.",
        "",
        "## Counts",
        "",
        "| | Series | Sites |",
        "| --- | --- | --- |",
        f"| **Shown (in both)** | **{len(both)}** | **{len(sites(both))}** |",
        f"| Model only — no 2025 data | {len(model_only)} | {len(sites(model_only))} |",
        f"| Actuals only — not in 2024 training | {len(actual_only)} | {len(sites(actual_only))} |",
        f"| Model total | {len(model_series)} | {len(sites(model_series))} |",
        f"| Actuals total | {len(actual_series)} | {len(sites(actual_series))} |",
        "",
        f"Of the {len(both)} shown, **{len(thin)}** recorded fewer than {THIN_DAYS}",
        "days in 2025. Those are flagged `thin` rather than excluded — the UI",
        "already explains a date with no recorded line, and dropping a real",
        "counter would hide coverage that exists.",
        "",
        "## Excluded — in the model, no 2025 data",
        "",
        "Forecastable, but nothing to score against. Most likely retired or",
        "offline through 2025.",
        "",
    ]
    if model_only:
        md += ["| Site | Series |", "| --- | --- |"]
        md += [f"| {p} | {', '.join(sorted(s))} |"
               for p, s in sorted(group(model_only).items())]
    else:
        md.append("_None._")

    md += [
        "",
        "## Excluded — recorded in 2025, absent from 2024 training",
        "",
        "The model **refuses** these: with no 2024 history there is no profile,",
        "and `forecast_dates()` raises rather than guessing. That is correct —",
        "a counter's traffic LEVEL cannot be inferred from its location (road",
        "class spans 0.1 to 1,351 veh/h), so any number would be invented.",
        "",
        "To forecast them, retrain with 2025 included.",
        "",
    ]
    if actual_only:
        md += ["| Site | Series |", "| --- | --- |"]
        md += [f"| {p} | {', '.join(sorted(s))} |"
               for p, s in sorted(group(actual_only).items())]
    else:
        md.append("_None._")

    if thin:
        md += ["", f"## Shown but thin (< {THIN_DAYS} recorded days in 2025)", "",
               "| Site | Series | Days |", "| --- | --- | --- |"]
        md += [f"| {i['poste_id']} | {i['direction']}-{i['vehicule']} | "
               f"{i['recorded_days_2025']} |" for i in
               sorted(thin, key=lambda i: i["recorded_days_2025"])]

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"included (both years) : {len(both)} series, {len(sites(both))} sites")
    print(f"excluded, model only  : {len(model_only)} series, {len(sites(model_only))} sites")
    print(f"excluded, actuals only: {len(actual_only)} series, {len(sites(actual_only))} sites")
    print(f"shown but thin        : {len(thin)}")
    print(f"\nwrote {OUT_JSON.name} and {OUT_MD.name}")


if __name__ == "__main__":
    main()
