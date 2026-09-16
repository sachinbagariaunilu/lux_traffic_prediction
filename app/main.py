"""Luxembourg traffic forecast API.

ONE FILE, TWO DEPLOYED SERVICES. SERVICE_ROLE decides which half this process
is -- which models it loads and which routes it registers. See the SERVICE_ROLE
block below for the memory measurements that forced the split, and
SPLIT_SERVICES.md for the routing table.

    SERVICE_ROLE=forecast   the LONG-HORIZON models, below. Answers a bare date.
    SERVICE_ROLE=lag        the SHORT-HORIZON models. The caller supplies recent
                            observed counts; more accurate, but only within the
                            model's lead. GET /forecast/{lead}h/spec, POST
                            /forecast/{lead}h.
    SERVICE_ROLE=all        both, in one process. Local use only -- it does not
                            fit a 512 MB instance.

The rest of this docstring is the forecast half.

Serves TWO models, chosen with ?model= on /counters, /manifest and /forecast:

    model=2024        trained on 2024 only. It has never seen 2025, so a 2025
                      forecast can be scored against actuals/ -- this is the
                      VALIDATION product, the one that proves the method works.
    model=2024_2025   trained on both years. More accurate on 2026 (11.4% vs
                      12.5% average error, measured against roadside sensors),
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
import os
import threading
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from forecast import predict

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# SERVICE ROLE -- which half of the API this process is
# --------------------------------------------------------------------------
#
# WHY THERE ARE TWO. Four bundles do not fit one 512 MB Render instance.
# MEASURED, uvicorn --workers 1, RSS after forcing every bundle to load:
#
#   SERVICE_ROLE=forecast   /forecast, /counters, /manifest, /actuals, /models
#                           models/forecast_model_2024.pkl and _2024_2025.pkl
#                           87 MB idle -> 257 MB one bundle -> 346 MB both
#   SERVICE_ROLE=lag        GET /forecast/{lead}h/spec and POST /forecast/{lead}h
#                           models/forecast_model_24h.pkl and _48h.pkl
#                           82 MB idle -> 256 MB one bundle -> 280 MB both
#
# One process holding all four is 346 + 280 - 85 (the shared interpreter and
# libraries, counted once) = ~540 MB, over the cap BEFORE request overhead.
# Render does not error on that -- it OOM-restarts, which reads as requests
# vanishing and a cold start rather than a fault, so it is the kind of failure
# that gets misdiagnosed for a week.
#
# Split, each service has genuine headroom: 166 MB spare on the forecast
# service, 232 MB on the lag service.
#
# Those figures are macOS, so treat them as the SHAPE rather than the exact
# Linux numbers -- but they agree with the 327 MB that get_bundle()'s comment
# below records for the two forecast bundles on the deployed platform, which is
# the one figure measured on Render itself.
#
# ONE codebase, not two repos. A fork would drift, and this file's header
# records what it cost the last time prediction logic lived in two places.
# Dockerfile builds the forecast image and Dockerfile.lag the lag image; each
# COPYs only its OWN bundles and bakes its role in as ENV, so a service cannot
# be started against the wrong models by forgetting an environment variable.
#
# SERVICE_ROLE=all is the old single-service behaviour, kept for local work
# where memory is not 512 MB. Do not deploy it to the free plan.
ROLE = os.environ.get("SERVICE_ROLE", "all").strip().lower()
if ROLE not in {"forecast", "lag", "all"}:
    raise RuntimeError(
        f"SERVICE_ROLE={ROLE!r} is not one of forecast, lag, all. This is "
        f"deliberately fatal at import: a typo that silently fell back to a "
        f"default would deploy a service serving the wrong half of the API.")
SERVES_FORECAST = ROLE in {"forecast", "all"}
SERVES_LAG = ROLE in {"lag", "all"}

# Where the other half lives, so a caller that hits the wrong service is TOLD
# where to go rather than getting a bare 404. Optional -- set it in render.yaml
# once both services have URLs. Splitting one API into two is exactly the
# change that breaks existing clients silently, and this is the antidote.
COMPANION_URL = os.environ.get("COMPANION_URL", "").strip() or None
if COMPANION_URL and "://" not in COMPANION_URL:
    # render.yaml fills this with fromService/property: host, which is a BARE
    # hostname. Pasting that straight into an error message gives the reader
    # something they cannot click.
    COMPANION_URL = f"https://{COMPANION_URL}"

app = FastAPI(title="Luxembourg Traffic Forecast", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# Every response is JSON and compresses hard: /counters 306 KB -> 36 KB.
# uvicorn compresses nothing of its own.
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _noop(fn):
    """Leave the handler defined but unrouted on a service that is not its role."""
    return fn


def forecast_get(*a, **kw):
    """@app.get, but only on an instance that holds the forecast bundles.

    Registering the route and then failing inside it would be worse: the route
    would appear in /docs and in the 404 handler's endpoint list, so a caller
    would believe this service could answer it.
    """
    return app.get(*a, **kw) if SERVES_FORECAST else _noop


def lag_get(*a, **kw):
    return app.get(*a, **kw) if SERVES_LAG else _noop


def lag_post(*a, **kw):
    return app.post(*a, **kw) if SERVES_LAG else _noop

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

# --------------------------------------------------------------------------
# The SHORT-HORIZON model -- deliberately NOT in MODEL_FILES
# --------------------------------------------------------------------------
#
# It is a different product, not a third variant of the same one, and putting
# it in MODEL_FILES would make four existing routes lie. /models, /counters,
# /manifest and /health all iterate MODEL_FILES and assume a manifest plus
# "can answer any date". This model can answer NO date without 31 days of
# observed counts supplied by the caller, and has no manifest at all.
#
# OPTIONAL AT IMPORT. The file may not be deployed -- scripts/deploy_to_backend.sh
# ships it only if it has been built -- so this loads lazily and the service
# starts fine without it. POST /forecast24h then returns 503 with the reason.
#
# MEMORY. app/main.py's get_bundle() comment records 327 MB resident for the two
# forecast bundles against Render's 512 MB cap, peaking at 512 while serving.
# This bundle is larger (28 features, not 16) at roughly 85 MB resident, which
# would put a three-bundle process near 410 MB before serving overhead. It is
# lazy for that reason: a deployment that never calls /forecast24h never pays
# for it. If this endpoint goes into real use, the instance needs resizing --
# measure before assuming otherwise.
LAG_MODELS = {24: "forecast_model_24h.pkl", 48: "forecast_model_48h.pkl"}
_LAG_BUNDLES: dict[int, dict] = {}
_LAG_LOCK = threading.Lock()

# MEMORY. This is why the service is split -- the SERVICE_ROLE block at the top
# of this file has the measurements. On a SERVICE_ROLE=lag instance both leads
# fit comfortably, 280 MB worst case against 512, so neither has to be dropped.
#
# Still LAZY, for two reasons that outlive the split: a service whose callers
# only ever ask for 48h never pays for the 24h bundle, and cold start does half
# the work, which a free plan pays for often.
#
# Do NOT add eviction. Dropping a bundle frees nothing -- Python returns the
# memory to its own allocator, not to the OS -- and it re-costs the load on
# every switch. That was measured when eviction was considered for the forecast
# bundles, and rejected.
#
# If this service ever must shed a lead, drop 24h and keep 48h: docs/LEAD_LAG.md
# 3 argues it costs ~0.5 MAE against the 24h model and keeps answering when a
# feed slips by a day, where the 24h model refuses outright.


def get_lag_bundle(lead: int) -> dict:
    """The short-horizon bundle for `lead`, loaded on first use. 503 if absent."""
    if lead not in _LAG_BUNDLES:
        fname = LAG_MODELS[lead]
        path = ROOT / "models" / fname
        if not path.exists():
            raise HTTPException(status_code=503, detail=(
                f"the {lead}h short-horizon model is not deployed on this "
                f"instance (no models/{fname}). The /forecast endpoints are "
                f"unaffected. Build it with scripts/train_lag24.py --lead "
                f"{lead} and redeploy."))
        with _LAG_LOCK:
            if lead not in _LAG_BUNDLES:
                _LAG_BUNDLES[lead] = predict.load_bundle(path)
    return _LAG_BUNDLES[lead]


def _require_lead(lead: int) -> None:
    """Only the leads we actually ship a bundle for."""
    if lead not in LAG_MODELS:
        raise HTTPException(status_code=404, detail=(
            f"no {lead}h model. Available short-horizon leads: "
            f"{sorted(LAG_MODELS)}. Lead is how many hours ahead the forecast "
            f"is issued, and it fixes how stale the supplied counts may be."))


# --------------------------------------------------------------------------
# Server-held recent history -- what makes GET /forecast/{lead}h possible
# --------------------------------------------------------------------------
#
# Built by scripts/build_lag_history.py as the last ~45 days of actuals/, one
# small file per counter. Read per request rather than held in memory, the same
# way /actuals does: 3.6 MB across 264 files, and a request touches one of them.
#
# WHAT THIS BUYS. Without it the caller had to send the counts itself, so our
# own frontend fetched history from the FORECAST service and POSTed 55 KB of it
# back to this one -- our data, moved between two of our own machines, through
# the user's connection. With it the same request is a bare GET.
#
# WHAT IT IS NOT. Not `data/live`, and it does not make these models generally
# servable. It is a frozen snapshot of the tail of the recording, so it can
# answer only dates within one lead of where the data stops -- 2026-01-01 and
# 2026-01-02 today. Anything later needs a real feed. See LEAD_LAG.md.
HISTORY = ROOT / "data" / "history"


def load_history(poste_id: int, direction: int, vehicule: str) -> pd.DataFrame:
    """This series' recent counts, as forecast_with_history() wants them.

    Raises HTTPException rather than returning empty: every failure here is a
    different thing the caller needs told apart -- no snapshot deployed at all,
    no file for this counter, no rows for this series.
    """
    if not HISTORY.is_dir():
        raise HTTPException(status_code=503, detail=(
            "no recent-history snapshot is deployed on this instance, so this "
            "endpoint cannot assemble the counts itself. Send them in the body "
            f"with POST /forecast/{{lead}}h instead, or rebuild the snapshot "
            "with scripts/build_lag_history.py."))
    path = HISTORY / f"{poste_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=(
            f"no recent history for counter {poste_id}. It may have stopped "
            f"reporting before the snapshot window; POST /forecast/{{lead}}h "
            f"with your own counts still works."))
    blob = json.loads(path.read_text())
    days = blob.get("series", {}).get(f"{direction}-{vehicule}")
    if not days:
        raise HTTPException(status_code=404, detail=(
            f"counter {poste_id} has no recent history for direction "
            f"{direction}, vehicle {vehicule!r}. Available: "
            f"{sorted(blob.get('series', {}))}"))
    rows = [(f"{day}T{hour:02d}:00:00", float(v))
            for day in sorted(days) for hour, v in enumerate(days[day])]
    return pd.DataFrame(rows, columns=["TIME_STAMP", "TRAFFIC_VOLUME"])


class HistoryPoint(BaseModel):
    t: str = Field(..., description="ISO hour, e.g. 2025-01-14T08:00:00")
    v: float = Field(..., description="observed vehicles in that hour")


class Forecast24hRequest(BaseModel):
    poste_id: int
    direction: int = Field(..., ge=1, le=2)
    vehicule: str = Field(..., pattern="^[VC]$")
    date: str = Field(..., description="target day, YYYY-MM-DD")
    history: list[HistoryPoint] = Field(..., description=(
        "observed hourly counts for THIS series, ending no more than 24h "
        "before the target day starts. See /forecast24h/spec."))


# What each model serves. Built by scripts/build_counter_manifest.py as EVERY
# series the model knows, with a per-series scoreable_2025 flag rather than the
# old intersection rule -- that rule would have left the 2024+2025 model serving
# nothing, since no year holds both its forecast and a recorded count.
# COUNTER_MANIFEST.md records the counts and every exclusion with its reason.
#
# ROLE-GUARDED. The lag service does not ship these files -- it serves no
# counter list and no scoreable flag -- and an unconditional read here would
# crash it at import with a FileNotFoundError that looks like a broken deploy
# rather than a service doing exactly what it should.
MANIFESTS = ({k: json.loads((ROOT / f"counter_manifest_{k}.json").read_text())
              for k in MODEL_FILES} if SERVES_FORECAST else {})
SERVED = {k: {(c["poste_id"], c["direction"], c["vehicule"]) for c in m["served"]}
          for k, m in MANIFESTS.items()}
# Which series have 2025 actuals, so /forecast can tell the UI up front whether
# the panel it is about to draw will have a second line.
SCOREABLE = {k: {(c["poste_id"], c["direction"], c["vehicule"])
                 for c in m["served"] if c["scoreable_2025"]}
             for k, m in MANIFESTS.items()}
MANIFEST = MANIFESTS[DEFAULT_MODEL] if SERVES_FORECAST else None
# The honest accuracy figure for the default model: the FULL unseen year,
# not the 46-day window, which has Christmas as 2 of its 46 days and
# overstated the last feature 4.6x. Recorded in the manifest by the builder.
EXPECTED_MAE = MANIFEST["holdout_mae"] if SERVES_FORECAST else None


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

# Split by role, because this list is what the 404 handler and / advertise. A
# service must never offer an endpoint it does not hold the model for.
COMMON_ENDPOINTS = [
    {"path": "/health", "method": "GET",
     "description": "service status, model accuracy, and coverage bounds"},
]

FORECAST_ENDPOINTS = [
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
]

LAG_ENDPOINTS = [
    {"path": "/forecast/{lead}h/spec", "method": "GET",
     "description": "how much history the short-horizon model needs, read off the bundle",
     "example": "/forecast/48h/spec"},
    {"path": "/forecast/{lead}h", "method": "GET",
     "description": ("hourly forecast using the recent counts this service "
                     "already holds -- no payload. Only dates within one lead "
                     "of where that snapshot ends can be answered"),
     "required_params": {
         "poste_id": "counter id, e.g. 1410",
         "direction": "1 or 2",
         "vehicule": "V for cars, C for trucks",
         "date": "YYYY-MM-DD"},
     "example": "/forecast/24h?poste_id=1410&direction=1&vehicule=V&date=2026-01-01"},
    {"path": "/forecast/{lead}h", "method": "POST",
     "description": ("same, but from counts YOU supply -- for a caller whose "
                     "feed is fresher than ours. lead is how many hours ahead "
                     "it is issued, and it fixes how stale those counts may be"),
     "required_body": {
         "poste_id": "counter id, e.g. 1410",
         "direction": "1 or 2",
         "vehicule": "V for cars, C for trucks",
         "date": "YYYY-MM-DD",
         "history": "[{t: ISO hour, v: vehicles}] -- see /forecast/{lead}h/spec"},
     "example": "POST /forecast/48h"},
]

ENDPOINTS = (COMMON_ENDPOINTS
             + (FORECAST_ENDPOINTS if SERVES_FORECAST else [])
             + (LAG_ENDPOINTS if SERVES_LAG else [])
             + [{"path": "/docs", "method": "GET",
                 "description": "interactive API docs"}])


ROLE_SUMMARY = {
    "forecast": ("long-horizon models. Any date from a bare calendar date, no "
                 "observed counts needed."),
    "lag": ("short-horizon models. Needs recent observed counts supplied in the "
            "request body; more accurate than the forecast models, but only "
            "within its lead."),
    "all": "both halves in one process. Local use only -- see SERVICE_ROLE.",
}


def _deployed_models() -> dict:
    """What THIS instance can actually load, by role."""
    if SERVES_FORECAST:
        return {k: {"trained_through": MANIFESTS[k]["trained_through"][:10],
                    "role": MANIFESTS[k]["role"]} for k in MODEL_FILES}
    return {f"{lead}h": {"role": "short-horizon, caller supplies history",
                         "deployed": (ROOT / "models" / f).exists()}
            for lead, f in LAG_MODELS.items()}


@app.get("/")
def root():
    return {"service": "Luxembourg Traffic Forecast", "version": "3.0",
            # Which half this is. A client that got the wrong URL finds out
            # here rather than from a 404 on the route it wanted.
            "service_role": ROLE,
            "serves": ROLE_SUMMARY[ROLE],
            "companion_service": COMPANION_URL,
            "models": _deployed_models(),
            "default_model": DEFAULT_MODEL if SERVES_FORECAST else None,
            "endpoints": ENDPOINTS}


@forecast_get("/models")
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
                     "roadside sensors -- 2024_2025 scores 11.4% average error "
                     "against 12.5% for 2024.")}


# Paths that exist on the OTHER service. Splitting one API in two breaks
# existing clients at exactly these paths, and a bare "Not Found" would send
# someone hunting for a bug in their own code.
_OTHER_ROLE_PATHS = {
    "forecast": ("/counters", "/manifest", "/models", "/actuals", "/forecast"),
    "lag": ("/forecast/",),          # /forecast/24h, /forecast/48h/spec
}


def _wrong_service_hint(path: str) -> str | None:
    """Say so when the path belongs to the half of the API we are not."""
    if not SERVES_LAG and path.startswith("/forecast/") and path.endswith(
            ("h", "h/spec")):
        where = COMPANION_URL or "the SERVICE_ROLE=lag service"
        return (f"{path} is a SHORT-HORIZON route and this instance is "
                f"SERVICE_ROLE={ROLE}. It lives on {where}.")
    if not SERVES_FORECAST and any(
            path == q or path.startswith(q.rstrip("/") + "/")
            for q in _OTHER_ROLE_PATHS["forecast"]):
        where = COMPANION_URL or "the SERVICE_ROLE=forecast service"
        return (f"{path} is a LONG-HORIZON route and this instance is "
                f"SERVICE_ROLE={ROLE}. It lives on {where}. This service "
                f"answers /forecast/24h and /forecast/48h only.")
    return None


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """On an unknown path, show what could have been asked for.

    Starlette uses the literal detail 'Not Found' when no route matches. A 404
    raised by a handler carries its own message and passes through untouched.
    """
    if exc.status_code == 404 and exc.detail == "Not Found":
        return JSONResponse(status_code=404, content={
            "error": f"Unknown path: {request.url.path}",
            "hint": (_wrong_service_hint(request.url.path)
                     or "check the spelling, or use one of the endpoints below"),
            "service_role": ROLE,
            "companion_service": COMPANION_URL,
            "endpoints": ENDPOINTS})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health():
    """Render's healthCheckPath. MUST answer on BOTH services.

    It therefore reads nothing role-specific before branching: the lag service
    has no manifest, so touching MANIFEST unconditionally here would fail the
    health check and Render would never mark the service live.
    """
    if not SERVES_FORECAST:
        return {
            "status": "ok",
            "service_role": ROLE,
            "companion_service": COMPANION_URL,
            "leads": {str(lead): {"file": f,
                                  "deployed": (ROOT / "models" / f).exists(),
                                  # Lazily loaded: false until something has
                                  # actually forecast with it.
                                  "loaded": lead in _LAG_BUNDLES}
                      for lead, f in LAG_MODELS.items()},
            "note": ("this service needs observed counts in the request body; "
                     "see /forecast/{lead}h/spec for how many hours"),
        }
    return {
        "status": "ok",
        "service_role": ROLE,
        "companion_service": COMPANION_URL,
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


@forecast_get("/counters")
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


@forecast_get("/manifest")
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


@forecast_get("/actuals/{poste_id}")
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


@forecast_get("/forecast")
def forecast(poste_id: int = Query(...), direction: int = Query(..., ge=1, le=2),
             vehicule: str = Query(..., pattern="^[VC]$"), date: str = Query(...),
             model: str = Query(DEFAULT_MODEL),
             growth_pct: float = Query(
                 0.0, ge=-5.0, le=5.0,
                 description="Compound annual growth applied to the prediction, "
                             "counted from the model's trained_through year. "
                             "DEFAULT 0 -- the model's own zero-growth answer. "
                             "Measured band is 0.0-1.0%/yr, central 0.5. Echoed "
                             "in the response so the assumption is never silent.")):
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
        # growth_pct is handed to predict.forecast_dates(), NOT applied here.
        # ONE implementation, in the vendored forecast package. This file's own
        # header records what duplicating prediction logic cost last time.
        out = predict.forecast_dates(poste_id, direction, vehicule, date,
                                     bundle=get_bundle(model), quiet=True,
                                     growth_pct_per_year=growth_pct)
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
        # Echoed even when zero, so a consumer can always tell which it got.
        "growth_pct_per_year": growth_pct,
        "growth_base_year": (int(out["growth_base_year"].iloc[0])
                             if "growth_base_year" in out else None),
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
            "June-July 2026 roadside sensors it averages 11.4% error, "
            "against 12.5% for model=2024."),
    }


@lag_get("/forecast/{lead}h/spec")
def forecast_lead_spec(lead: int):
    """What POST /forecast24h needs, read off the bundle rather than described.

    A caller that has to guess the history window will guess wrong, and the
    failure mode -- a forecast built on too-short history -- looks like a
    working forecast. So the requirement is served, not documented.
    """
    _require_lead(lead)
    b = get_lag_bundle(lead)
    return {
        "kind": b["kind"],
        "lead_hours": b["lead"],
        "history_hours": b["history_hours"],
        "history_days": round(b["history_hours"] / 24, 1),
        "lag_ladder": b["lag_ladder"],
        "trained_through": b["trained_through"],
        "profiles_built_from": b["profiles_built_from"],
        "blind_test_scores": b.get("scores"),
        "rule": (
            f"history must cover {b['history_hours']}h for this series and end "
            f"no earlier than {b['lead']}h before the first target hour. Gaps "
            f"are fine; staleness is refused."),
    }


def _lag_forecast(lead: int, b: dict, poste_id: int, direction: int,
                  vehicule: str, date: str, hist: pd.DataFrame,
                  history_source: str) -> dict:
    """The answer, shared by the GET and POST forms.

    ONE implementation on purpose. The two routes differ only in where the
    counts came from, and the moment that difference is allowed to fork the
    response-building the two will drift -- which is the exact failure this
    file's header records from the last time prediction logic was duplicated.
    """
    try:
        out = predict.forecast_with_history(
            poste_id, direction, vehicule, date, history=hist, bundle=b, quiet=True)
    except ValueError as ex:
        # Stale history, too little history, unknown series. All three are about
        # the DATA, not the URL -- 422 whichever route asked.
        raise HTTPException(status_code=422, detail=str(ex))

    # PARITY WITH /forecast, deliberately. The UI routes a date to whichever
    # model can answer it and renders ONE report either way, so a field present
    # on one response and absent on the other is not a cosmetic gap -- it is a
    # panel that loses its holiday chip precisely on 1 January, the date these
    # models exist to answer.
    holiday = bool((out["is_public_holiday"] | out["is_school_holiday"]).any())
    return {
        "counter": {"poste_id": poste_id, "direction": direction,
                    "vehicule": vehicule},
        "model": f"{lead}h",
        "kind": b["kind"],
        "date": date,
        # Which of the two ways the counts arrived. Worth stating: a caller that
        # believes it sent its own fresh counts, and is silently being answered
        # from a frozen snapshot, would misread the result.
        "history_source": history_source,
        # The last hour of real traffic behind this forecast. The UI states it
        # ("given real counts to 31 December"), and with the GET form the caller
        # no longer holds the history itself, so it cannot work this out.
        "history_through": str(pd.to_datetime(hist["TIME_STAMP"]).max())[:19],
        "history_hours_supplied": len(hist),
        "history_hours_required": b["history_hours"],
        "is_holiday_period": holiday,
        "daily_total": round(float(out["PREDICTED"].sum())),
        "hourly": [{"hour": int(t.hour), "predicted": float(p),
                    "typical_for_slot": float(n)}
                   for t, p, n in zip(out["TIME_STAMP"], out["PREDICTED"],
                                      out["typical_for_slot"])],
    }


@lag_get("/forecast/{lead}h")
def forecast_lead_get(lead: int, poste_id: int = Query(...),
                      direction: int = Query(..., ge=1, le=2),
                      vehicule: str = Query(..., pattern="^[VC]$"),
                      date: str = Query(...)):
    """Short-horizon forecast using the counts THIS SERVICE already holds.

    The same answer as POST /forecast/{lead}h, with no payload: the service
    reads the recent history itself. That is the whole point -- the POST form
    made our own frontend fetch 55 KB of our own data and send it back to us.

    Only dates within one lead of where the snapshot ends can be answered, so
    this is 2026-01-01 and 2026-01-02 today. Later dates get a 422 naming how
    stale the snapshot is, which is the honest answer until a feed exists.

    POST remains the right call for anyone whose own counts are fresher than
    ours -- an operator with a live feed.
    """
    _require_lead(lead)
    b = get_lag_bundle(lead)
    hist = load_history(poste_id, direction, vehicule)
    return _lag_forecast(lead, b, poste_id, direction, vehicule, date, hist,
                         history_source="server snapshot")


@lag_post("/forecast/{lead}h")
def forecast_lead(lead: int, req: Forecast24hRequest = Body(...)):
    """Short-horizon forecast from counts the CALLER supplies.

    Use this when your own counts are fresher than ours -- an operator with a
    live feed. When they are not, GET /forecast/{lead}h is the same answer with
    no payload.

    Three refusals, all deliberate:
      422  history too short, too stale, or the series is unknown to the model
      503  the short-horizon bundle is not deployed on this instance
      404  never -- an unknown series is a 422 here, because the request body
           is what is wrong, not the URL
    """
    _require_lead(lead)
    b = get_lag_bundle(lead)
    if not req.history:
        raise HTTPException(status_code=422, detail=(
            f"history is empty. This model needs {b['history_hours']}h of "
            f"observed counts; see GET /forecast/{lead}h/spec. If you have no "
            f"counts of your own, GET /forecast/{lead}h uses ours."))

    hist = pd.DataFrame({"TIME_STAMP": [h.t for h in req.history],
                         "TRAFFIC_VOLUME": [h.v for h in req.history]})
    return _lag_forecast(lead, b, req.poste_id, req.direction, req.vehicule,
                         req.date, hist, history_source="caller-supplied")
