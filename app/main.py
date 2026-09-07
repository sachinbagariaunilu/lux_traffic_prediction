"""Luxembourg traffic forecast API.

Serves TWO models, chosen with ?model= on /counters, /manifest and /forecast:

    model=2024        trained on 2024 only. It has never seen 2025, so a 2025
                      forecast can be scored against actuals/ -- this is the
                      VALIDATION product, the one that proves the method works.
    model=2024_2025   trained on both years. More accurate on 2026 (13.3% vs
                      14.4% average error, measured against roadside sensors),
                      but it trained on 2025 and so cannot be honestly scored
                      on any date up to 2025-12-31. FORECASTING product.

That asymmetry is enforced, not documented: /forecast REFUSES a date inside a
model's own training range and says why. Otherwise the 2024_2025 model could be
plotted against 2025 actuals and would look excellent for the wrong reason.

Every series a model knows is served. A series without 2025 actuals is still
forecastable -- the manifest flags it `scoreable_2025: false` so the UI can say
"no recorded data to compare" instead of hiding the counter, which reads as a
bug rather than a data boundary.

ARCHITECTURE -- three files, one source of truth each:

    forecast/            the prediction code, VENDORED from the training repo.
                         Do not edit here. Copy it over when the model changes,
                         so the API and the training pipeline can never drift.
    models/*.pkl         the model bundle.
    data/external/*.json the calendar, read at CALL TIME. Extending the holiday
                         file to 2030 needs no retraining and no redeploy of
                         the .pkl.
    counter_manifest_<model>.json
                         which series each model serves, and which of them have
                         2025 actuals to be scored against. Derived from the
                         bundles -- rebuild it whenever a model changes.

The previous version reimplemented feature construction in app/forecast.py --
75 lines duplicating forecast/predict.py. It silently built only 12 of the
model's 14 features and served expected_mae=18.3, a figure that belonged to a
different model variant entirely. Both classes of bug disappear when there is
one implementation.

NOTE the bundle pickles a forecast.features.Profiles dataclass, so `forecast`
must be importable before joblib.load(). uvicorn puts the app root on sys.path,
which is why `from forecast import predict` works here; a bare script needs to
insert it (see scripts/build_counter_manifest.py).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from forecast import predict

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Luxembourg Traffic Forecast", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])
# Every response is JSON and compresses hard: /counters 306 KB -> 36 KB.
# uvicorn compresses nothing of its own.
app.add_middleware(GZipMiddleware, minimum_size=1000)

MODEL_FILES = {"2024": "forecast_model_2024.pkl",
               "2024_2025": "forecast_model_2024_2025.pkl"}
DEFAULT_MODEL = "2024"

# LAZILY loaded, then cached for the life of the process. Never per request.
#
# Loading both at import cost 327 MB resident against Render's 512 MB free-plan
# cap -- measured, and serving requests then peaked at 512. A bundle is 29 MB on
# disk and ~70 MB in memory, so deferring the second one keeps the floor at
# 259 MB and halves cold-start work, which a free plan pays often.
#
# What this does NOT do is cap memory: once someone asks for a 2026 date the
# second bundle loads and stays, and the total is the same 327 MB. Eviction was
# measured and rejected -- dropping one bundle freed nothing (Python returns
# memory to its own allocator, not the OS) and it re-cost 135 ms on every
# switch between a 2025 and a 2026 date, which is the page's main interaction.
#
# EVERYTHING ELSE READS THE MANIFEST, NOT THE BUNDLE. /models, /health,
# /counters, /manifest and the trained-on-this-date guard all answer from
# counter_manifest_<model>.json, which the builder fills with the bundle's own
# metadata. Otherwise the guard would unpickle 29 MB just to refuse a request,
# and probing with bad dates would load both models.
_BUNDLES: dict[str, dict] = {}
_LOAD_LOCK = threading.Lock()


def get_bundle(model: str) -> dict:
    """The bundle for `model`, loading it on first use.

    The lock matters: the route handlers are sync `def`, so Starlette runs them
    in a threadpool and two concurrent first-requests would otherwise both
    unpickle 29 MB -- 140 MB of transient duplicate at the worst moment.
    """
    if model not in _BUNDLES:
        with _LOAD_LOCK:
            if model not in _BUNDLES:          # re-check: another thread may have won
                _BUNDLES[model] = predict.load_bundle(
                    ROOT / "models" / MODEL_FILES[model])
    return _BUNDLES[model]

# What each model serves. Built by scripts/build_counter_manifest.py as EVERY
# series the model knows, with a per-series scoreable_2025 flag rather than the
# old intersection rule -- that rule would have left the 2024+2025 model serving
# nothing, since no year holds both its forecast and a recorded count.
# COUNTER_MANIFEST.md records the counts and every exclusion with its reason.
MANIFESTS = {k: json.loads((ROOT / f"counter_manifest_{k}.json").read_text())
             for k in MODEL_FILES}
SERVED = {k: {(c["poste_id"], c["direction"], c["vehicule"]) for c in m["served"]}
          for k, m in MANIFESTS.items()}
# Which series have 2025 actuals, so /forecast can tell the UI up front whether
# the panel it is about to draw will have a second line.
SCOREABLE = {k: {(c["poste_id"], c["direction"], c["vehicule"])
                 for c in m["served"] if c["scoreable_2025"]}
             for k, m in MANIFESTS.items()}
MANIFEST = MANIFESTS[DEFAULT_MODEL]
# The honest accuracy figure for the default model: the FULL unseen year,
# not the 46-day window, which has Christmas as 2 of its 46 days and
# overstated the last feature 4.6x. Recorded in the manifest by the builder.
EXPECTED_MAE = MANIFEST["holdout_mae"]


def _pick(model: str) -> str:
    """Validate the ?model= parameter, listing the choices on a bad one."""
    if model not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=(
            f"unknown model {model!r}. Choose one of: "
            f"{', '.join(sorted(MODEL_FILES))}"))
    return model


def _trained_range(model: str) -> tuple[str, str]:
    """First and last date the model fitted on, as YYYY-MM-DD.

    From the manifest, deliberately: this is called on every /forecast to decide
    whether to refuse, and reading the bundle here would load 29 MB just to say
    no.
    """
    m = MANIFESTS[model]
    return str(m["profiles_built_from"]).split("..")[0], str(m["trained_through"])[:10]


def _refuse_if_trained_on(model: str, date: str) -> None:
    """Refuse a forecast for a date the model was fitted on.

    A model scored on its own training data reports an accuracy it does not
    have. The training repo guards this with assert_no_leakage() and
    confirm_holdout_spend; this is the same rule at the API boundary, where a
    UI could otherwise plot model=2024_2025 against 2025 actuals and publish a
    flattering number nobody could reproduce.
    """
    start, end = _trained_range(model)
    if start <= date <= end:
        other = next((k for k in MODEL_FILES
                      if not _trained_range(k)[0] <= date <= _trained_range(k)[1]),
                     None)
        raise HTTPException(status_code=409, detail=(
            f"model {model!r} trained on {start}..{end}, so {date} is inside its "
            f"own training data -- a forecast there would be memory, not "
            f"prediction, and scoring it would overstate the model."
            + (f" Use model={other!r} for this date." if other else "")))

# Recorded 2025 counts, one file per counter. Read from disk per request rather
# than held in memory: ~30 MB most callers never ask for.
ACTUALS = ROOT / "actuals"

ENDPOINTS = [
    {"path": "/health", "method": "GET",
     "description": "service status, model accuracy, and coverage bounds"},
    {"path": "/models", "method": "GET",
     "description": "the two models, what each is for, and which dates each may forecast"},
    {"path": "/counters", "method": "GET",
     "description": "every series the model can forecast, flagged for whether 2025 actuals exist",
     "optional_params": {"model": "2024 (default) or 2024_2025"}},
    {"path": "/manifest", "method": "GET",
     "description": "what is served, what cannot be scored, and why",
     "optional_params": {"model": "2024 (default) or 2024_2025"}},
    {"path": "/forecast", "method": "GET",
     "description": "hourly forecast for one counter on one date",
     "required_params": {
         "poste_id": "counter id, e.g. 1410 (see /counters)",
         "direction": "1 or 2",
         "vehicule": "V for cars, C for trucks",
         "date": "YYYY-MM-DD"},
     "optional_params": {"model": "2024 (default) or 2024_2025"},
     "example": "/forecast?poste_id=1410&direction=1&vehicule=V&date=2025-03-12"},
    {"path": "/actuals/{poste_id}", "method": "GET",
     "description": "what the counter actually recorded, hour by hour, 2025 only",
     "example": "/actuals/1410"},
    {"path": "/docs", "method": "GET", "description": "interactive API docs"},
]


@app.get("/")
def root():
    return {"service": "Luxembourg Traffic Forecast", "version": "3.0",
            "models": {k: {"trained_through": MANIFESTS[k]["trained_through"][:10],
                           "role": MANIFESTS[k]["role"]} for k in MODEL_FILES},
            "default_model": DEFAULT_MODEL,
            "endpoints": ENDPOINTS}


@app.get("/models")
def models():
    """The two models, and the dates each one may honestly forecast.

    `forecastable_from` is the day after a model's training data ends. Asking
    for anything earlier gets a 409 from /forecast, because that date is inside
    the model's own training set.
    """
    import datetime as _dt
    out = []
    for k in MODEL_FILES:
        m = MANIFESTS[k]
        start, end = _trained_range(k)
        nxt = (_dt.date.fromisoformat(end) + _dt.timedelta(days=1)).isoformat()
        out.append({
            "model": k,
            "role": m["role"],
            "trained_on": f"{start}..{end}",
            "forecastable_from": nxt,
            "calendar_through": m["calendar_through"],
            "series_served": m["counts"]["served"],
            "scoreable_2025": m["counts"]["scoreable_2025"],
            "scoreable_years": m["scoreable_years"],
            "full_year_mae": m["holdout_mae"],
            "blind_test_mae": next(
                (sc["MAE"] for sc in m["blind_test_scores"]
                 if sc["Model"].startswith("model")), None),
            "blind_test_window": ("Nov-Dec 2024" if k == "2024" else "Nov-Dec 2025"),
            "loaded": k in _BUNDLES,
        })
    return {"default": DEFAULT_MODEL, "models": out,
            "note": ("blind_test_mae is each model's own 46-day measurement and the "
                     "windows DIFFER, so the two numbers are not comparable. On the "
                     "one test both models face identically -- June-July 2026 "
                     "roadside sensors -- 2024_2025 scores 13.3% average error "
                     "against 14.4% for 2024.")}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """On an unknown path, show what could have been asked for.

    Starlette uses the literal detail 'Not Found' when no route matches. A 404
    raised by a handler carries its own message and passes through untouched.
    """
    if exc.status_code == 404 and exc.detail == "Not Found":
        return JSONResponse(status_code=404, content={
            "error": f"Unknown path: {request.url.path}",
            "hint": "check the spelling, or use one of the endpoints below",
            "endpoints": ENDPOINTS})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "trained_through": MANIFEST["trained_through"],
        "features": MANIFEST["features"],
        # Full unseen year, NOT the 46-day window. See EXPECTED_MAE above.
        "expected_mae": EXPECTED_MAE,
        "accuracy_basis": "full unseen year 2025",
        "series_served": len(SERVED[DEFAULT_MODEL]),
        "models": {k: {"trained_through": MANIFESTS[k]["trained_through"][:10],
                       "series_served": MANIFESTS[k]["counts"]["served"],
                       # Lazily loaded: false until something actually forecasts
                       # with it. Useful for seeing what the worker is holding.
                       "loaded": k in _BUNDLES}
                   for k in MODEL_FILES},
        # The UI must not offer dates past this -- beyond it the three calendar
        # features would flatten to constants and every Tuesday would return an
        # identical total, January the same as August.
        "calendar_from": MANIFEST["calendar_from"],
        "calendar_through": MANIFEST["calendar_through"],
        "scored_through": MANIFEST.get("scored_through", "2025-12-31"),
    }


@app.get("/counters")
def counters(model: str = Query(DEFAULT_MODEL)):
    """Every series this model can forecast, busiest first.

    `scoreable_2025` is the field the UI needs: false means the series recorded
    no 2025 hours, so a 2025 forecast will have no actual line to sit beside.
    The counter is still listed and still forecastable -- omitting it would look
    like a missing counter rather than missing ground truth.

    `avg_per_hour` covers the hours the counter actually reported, not the
    calendar year -- no counter reported every day.
    """
    model = _pick(model)
    m = MANIFESTS[model]
    out = []
    for c in m["served"]:
        veh = "cars" if c["vehicule"] == "V" else "trucks"
        out.append({**c,
                    "label": f"{c['route']} — {c['localite']} "
                             f"(dir {c['direction']}, {veh})"})
    return {"model": model, "count": len(out),
            "scoreable_2025": m["counts"]["scoreable_2025"],
            "not_scoreable_2025": m["counts"]["not_scoreable_2025"],
            "counters": out}


@app.get("/manifest")
def manifest(model: str = Query(DEFAULT_MODEL)):
    """What is served, what cannot be scored, and why.

    Exposed so the UI can state its own coverage instead of implying the network
    is smaller than it is.
    """
    model = _pick(model)
    m = MANIFESTS[model]
    return {"model": model, "role": m["role"], "rule": m["rule"],
            "trained_through": m["trained_through"],
            "scoreable_years": m["scoreable_years"],
            "counts": m["counts"],
            "not_scoreable_2025": m["not_scoreable_2025"],
            "excluded_actuals_only": m["excluded_actuals_only"]}


@app.get("/actuals/{poste_id}")
def actuals(poste_id: int):
    """What this counter actually recorded, hour by hour, for 2025.

    The counterpart to /forecast: that says what the model expects, this says
    what the road saw, and the difference is the only honest score.

    Shape -- {"poste_id": 1410, "series": {"1-V": {"2025-02-15": [24 ints]}}}.
    Sent straight off disk; the file is already the response body.
    """
    path = ACTUALS / f"{poste_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"No recorded 2025 data for counter {poste_id}")
    return FileResponse(path, media_type="application/json")


@app.get("/forecast")
def forecast(poste_id: int = Query(...), direction: int = Query(..., ge=1, le=2),
             vehicule: str = Query(..., pattern="^[VC]$"), date: str = Query(...),
             model: str = Query(DEFAULT_MODEL)):
    """Hourly forecast for one series on one date, from the chosen model.

    Two refusals, both deliberate:

      404  the model has no history for this series, so there is no profile and
           forecast_dates() would have to invent a level.
      409  the date is inside the model's own training range. That is not a
           forecast, and scoring it would overstate the model. The message
           names the other model, which can answer honestly.

    `scoreable` in the response says whether actuals/ has 2025 data for this
    series. False is not an error: the forecast is real, there is simply nothing
    recorded to plot against it.
    """
    model = _pick(model)
    _refuse_if_trained_on(model, date)
    key = (poste_id, direction, vehicule)
    if key not in SERVED[model]:
        served_by = [k for k in MODEL_FILES if key in SERVED[k]]
        raise HTTPException(status_code=404, detail=(
            f"counter ({poste_id}, {direction}, '{vehicule}') is not in model "
            f"{model!r} -- it has no history there, so the model refuses to "
            f"guess a level."
            + (f" Served by model={served_by[0]!r}." if served_by else
               " See /manifest for what is available.")))

    try:
        out = predict.forecast_dates(poste_id, direction, vehicule, date,
                                     bundle=get_bundle(model), quiet=True)
    except ValueError as ex:
        # Raised by the calendar-coverage guard or an unknown series. Both are
        # deliberate refusals, and the message explains which.
        raise HTTPException(status_code=404, detail=str(ex))

    holiday = bool((out["is_public_holiday"] | out["is_school_holiday"]).any())
    scoreable = key in SCOREABLE[model] and date.startswith("2025")
    return {
        "counter": {"poste_id": poste_id, "direction": direction,
                    "vehicule": vehicule},
        "model": model,
        "date": date,
        # Whether /actuals/{poste_id} can supply a recorded line for this date.
        # The UI shows the forecast either way and says so when it cannot score.
        "scoreable": scoreable,
        "scoreable_note": (
            None if scoreable else
            "no recorded 2025 data for this series -- the forecast stands, but "
            "there is nothing measured to compare it against"
            if key not in SCOREABLE[model] else
            "actuals/ covers 2025 only, so this date cannot be scored"),
        "hourly": [{"hour": int(t.hour), "predicted": float(p),
                    "typical_2024": float(n)}
                   for t, p, n in zip(out["TIME_STAMP"], out["PREDICTED"],
                                      out["typical_for_slot"])],
        "daily_total": round(float(out["PREDICTED"].sum())),
        "is_holiday_period": holiday,
        # One honest figure, from the full unseen year. The old API returned a
        # separate inflated number for holiday dates, taken from a different
        # model variant.
        "expected_error": (EXPECTED_MAE if model == DEFAULT_MODEL else None),
        "expected_error_note": (
            None if model == DEFAULT_MODEL else
            "no full-unseen-year figure exists for this model: it trained on "
            "2025, so there is no year it has not seen. Measured against "
            "June-July 2026 roadside sensors it averages 13.3% error, "
            "against 14.4% for model=2024."),
    }
