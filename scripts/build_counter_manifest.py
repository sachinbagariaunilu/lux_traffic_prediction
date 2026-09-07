"""Build one manifest per model: every series it can forecast, flagged for
whether that forecast can be SCORED.

    python scripts/build_counter_manifest.py

THE RULE CHANGED on 2026-09-07, and the change matters.

The old rule kept the INTERSECTION of the model and the 2025 actuals, on the
argument that a series without actuals produces "a panel with half the answer".
That was right while there was one model and one comparison. It is wrong now,
because there are two products:

    2024 model        forecasts 2025, and 2025 actuals exist -> SCOREABLE
    2024+2025 model   forecasts 2026 onward, no actuals exist -> forecast only

Under the intersection rule the second product would serve nothing at all: it
has no year with both a forecast and a recorded count. So the manifest now
serves EVERY series the model knows and carries a per-series flag:

    scoreable_2025   the series also recorded 2025 hours, so a forecast for a
                     2025 date can be plotted against what the road saw.

The UI lists all of them and says so when the comparison is unavailable, rather
than the counter silently not existing. A missing actual is a data boundary; a
missing counter looks like a bug.

Writes, per model:
    counter_manifest_<model>.json    what the API serves
    COUNTER_MANIFEST.md              the human-readable log, both models

Re-run whenever a model or actuals/ changes. Outputs are committed, so a change
in what the UI offers shows up in a diff rather than being discovered.
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

# The two products. Keys are the `model` parameter the API accepts.
MODELS = {
    "2024": {
        "file": "forecast_model_2024.pkl",
        "role": "validation -- forecasts a year it never saw, so it can be scored",
        "scoreable_years": [2025],
    },
    "2024_2025": {
        "file": "forecast_model_2024_2025.pkl",
        "role": "forecasting -- most recent data, no scoreable year available",
        "scoreable_years": [],
    },
}
ACTUALS = ROOT / "actuals"
OUT_MD = ROOT / "COUNTER_MANIFEST.md"

# A series with only a handful of recorded days can still be shown, but almost
# every date the user picks will have no actual line. Flagged, not excluded.
THIN_DAYS = 30


def read_actuals() -> dict[tuple[int, int, str], int]:
    """(poste, direction, vehicule) -> number of recorded 2025 days."""
    out: dict[tuple[int, int, str], int] = {}
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
            out[(pid, int(direction), vehicule)] = len(days)
    return out


def series_of(spec: dict) -> set:
    """The (poste, direction, vehicule) keys one bundle can forecast."""
    prof = joblib.load(ROOT / "models" / spec["file"])["profiles"]
    return {(int(r.POSTE_ID), int(r.DIRECTION), str(r.VEHICULE))
            for r in prof.provenance.itertuples()}


def build(model_key: str, spec: dict, actual_series: dict,
          all_model_series: dict) -> dict:
    bundle = joblib.load(ROOT / "models" / spec["file"])
    prof = bundle["profiles"]
    model_series = {
        (int(r.POSTE_ID), int(r.DIRECTION), str(r.VEHICULE))
        for r in prof.provenance.itertuples()
    }

    # Route / locality / coordinates. NOT in the bundle -- it carries only what
    # the model reads, and the model never sees them. Exported alongside so the
    # API can label a counter without the training repo present.
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
        # Mean over the hours the counter actually reported, not the calendar
        # year -- no counter reported every day.
        avg = float(prof.series.loc[
            (prof.series.POSTE_ID == p) & (prof.series.DIRECTION == d)
            & (prof.series.VEHICULE == v), "PROF_MEAN"].iloc[0])
        recorded = actual_series.get(key)
        return {
            "poste_id": p, "direction": d, "vehicule": v,
            "route": a.get("route"), "localite": a.get("localite"),
            "sens": a.get("sens"),
            # LUREF (EPSG:2169) METRES, as recorded in the source CSV.
            # NOT lat/lon -- reproject before putting on a map.
            "coord_x": a.get("coord_x"), "coord_y": a.get("coord_y"),
            "avg_per_hour": round(avg, 1),
            "days_reported": int(pr.N_OBS // 24) if pr is not None else None,
            "first_day": str(pr.FIRST_SEEN.date()) if pr is not None else None,
            "last_day": str(pr.LAST_SEEN.date()) if pr is not None else None,
            # The flag the UI needs. None means the series recorded no 2025
            # hours at all, so a 2025 forecast has nothing to be plotted against.
            "recorded_days_2025": recorded,
            "scoreable_2025": recorded is not None,
            # Which models can forecast this series at all. The UI needs it
            # because the model is chosen by the DATE: a series the 2024 model
            # has never seen cannot be forecast for 2025 however much 2026 data
            # exists for it, and the picker has to say so rather than let the
            # request 404.
            "served_by": sorted(k for k, ks in all_model_series.items()
                                if key in ks),
            "thin": recorded is not None and recorded < THIN_DAYS,
        }

    served = [entry(k) for k in sorted(model_series)]
    served.sort(key=lambda e: e["avg_per_hour"], reverse=True)
    scoreable = [e for e in served if e["scoreable_2025"]]
    unscoreable = [e for e in served if not e["scoreable_2025"]]
    actuals_only = sorted(set(actual_series) - model_series)

    payload = {
        "model": model_key,
        "model_file": spec["file"],
        "role": spec["role"],
        "trained_through": str(bundle["trained_through"]),
        "profiles_built_from": str(bundle["profiles_built_from"]),
        "rule": ("every series the model can forecast; scoreable_2025 says "
                 "whether 2025 actuals exist to compare against"),
        # Bundle metadata the API needs for /models, /health and the
        # trained-on-this-date guard. Recorded HERE so the API can answer all of
        # those, and refuse a bad date, without unpickling a 29 MB bundle -- the
        # bundles are lazily loaded and only an actual forecast should trigger
        # one. Regenerated from the bundle on every deploy, so it cannot drift.
        "features": len(bundle["features"]),
        "calendar_from": str(bundle["calendar_from"].date()),
        "calendar_through": str(bundle["calendar_through"].date()),
        "blind_test_scores": bundle["scores"],
        "holdout_mae": next(
            (sc["MAE"] for sc in ((bundle.get("holdout") or {}).get("scores") or [])
             if sc["Model"].startswith("model")), None),
        "scoreable_years": spec["scoreable_years"],
        "thin_days_threshold": THIN_DAYS,
        "counts": {
            "served": len(served),
            "scoreable_2025": len(scoreable),
            "not_scoreable_2025": len(unscoreable),
            "sites": len({e["poste_id"] for e in served}),
            "actuals_only": len(actuals_only),
        },
        "served": served,
        "not_scoreable_2025": [
            {"poste_id": e["poste_id"], "direction": e["direction"],
             "vehicule": e["vehicule"],
             "reason": "no 2025 recorded data -- forecastable but not scoreable"}
            for e in unscoreable],
        "excluded_actuals_only": [
            {"poste_id": p, "direction": d, "vehicule": v,
             "reason": "not in this model's training data -- the model refuses it"}
            for p, d, v in actuals_only],
    }
    out = ROOT / f"counter_manifest_{model_key}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}: {len(served)} served, "
          f"{len(scoreable)} scoreable on 2025, {len(unscoreable)} not")
    return payload


def main() -> None:
    actual_series = read_actuals()
    print(f"actuals/: {len(actual_series)} series with recorded 2025 days")
    all_model_series = {k: series_of(spec) for k, spec in MODELS.items()}
    for k, ks in all_model_series.items():
        print(f"  {k}: {len(ks)} series")
    built = {k: build(k, spec, actual_series, all_model_series)
             for k, spec in MODELS.items()}

    md = [
        "# Counter manifest",
        "",
        "What each model offers, and where a forecast can be scored.",
        "Generated by `scripts/build_counter_manifest.py`.",
        "",
        "## The rule",
        "",
        "Every series a model can forecast is **served**. A separate flag says",
        "whether it can also be **scored**:",
        "",
        "| Flag | Meaning |",
        "| --- | --- |",
        "| `scoreable_2025: true` | the series recorded 2025 hours, so a 2025 forecast can be plotted against what the road saw |",
        "| `scoreable_2025: false` | no 2025 recorded data — the forecast still works, there is just nothing to compare it to |",
        "",
        "This replaced an intersection rule that served only series present in",
        "both the model and the actuals. That rule made sense for one model and",
        "one comparison; with two products it would have left the 2024+2025",
        "model serving nothing, since no year has both its forecast and a",
        "recorded count.",
        "",
        "## The two models",
        "",
        "| Model | Trained through | Served | Scoreable on 2025 | Role |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k, p in built.items():
        md.append(f"| `{k}` | {p['trained_through'][:10]} | "
                  f"**{p['counts']['served']}** | {p['counts']['scoreable_2025']} | "
                  f"{p['role']} |")
    md += [
        "",
        "The 2024+2025 model knows more series because counters installed during",
        "2025 have no 2024 history and so cannot appear in a 2024-only model.",
        "",
    ]
    for k, p in built.items():
        ns = p["not_scoreable_2025"]
        md += [
            f"## `{k}` — not scoreable on 2025 ({len(ns)} series)",
            "",
            "Forecastable, nothing to score against. Most are counters retired or",
            "offline through 2025.",
            "",
        ]
        if ns:
            grouped = defaultdict(list)
            for e in ns:
                grouped[e["poste_id"]].append(f"{e['direction']}-{e['vehicule']}")
            md += ["| Site | Series |", "| --- | --- |"]
            md += [f"| {pid} | {', '.join(sorted(v))} |"
                   for pid, v in sorted(grouped.items())]
        else:
            md.append("_None._")
        md.append("")
        ao = p["excluded_actuals_only"]
        md += [
            f"## `{k}` — recorded in 2025, absent from this model ({len(ao)} series)",
            "",
            "The model **refuses** these: with no history there is no profile, and",
            "`forecast_dates()` raises rather than guessing. A counter's traffic",
            "LEVEL cannot be inferred from its location, so any number would be",
            "invented.",
            "",
        ]
        if ao:
            grouped = defaultdict(list)
            for e in ao:
                grouped[e["poste_id"]].append(f"{e['direction']}-{e['vehicule']}")
            md += ["| Site | Series |", "| --- | --- |"]
            md += [f"| {pid} | {', '.join(sorted(v))} |"
                   for pid, v in sorted(grouped.items())]
        else:
            md.append("_None._")
        md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
