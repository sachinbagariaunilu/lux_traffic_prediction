from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import joblib

from app.forecast import forecast_dates

app = FastAPI(title='Luxembourg Traffic Forecast', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'],
                   allow_headers=['*'])
# Every response here is JSON and compresses hard: /counters 306 KB -> 36 KB,
# /actuals 152 KB -> 58 KB on the busiest counter and ~112 KB -> 37 KB on a
# typical one. Without this the API serves them raw -- uvicorn compresses
# nothing of its own, unlike the static host /actuals used to be served from.
app.add_middleware(GZipMiddleware, minimum_size=1000)

B = joblib.load('models/forecast_model_2024.pkl')

# Recorded hourly counts, one file per counter, written by
# scripts/build_actuals.py. 2025 only -- 2024 is the training year, so the model
# reproduces it rather than forecasting it. Read from disk per request rather
# than held in memory: 30 MB of counts that most callers never ask for.
ACTUALS = Path('actuals')

ENDPOINTS = [
    {"path": "/health", "method": "GET",
     "description": "service status and which data the model was trained on"},
    {"path": "/counters", "method": "GET",
     "description": "every counter the model can forecast, busiest first"},
    {"path": "/forecast", "method": "GET",
     "description": "hourly forecast for one counter on one date",
     "required_params": {
         "poste_id": "counter id, e.g. 1410 (see /counters)",
         "direction": "1 or 2",
         "vehicule": "V for cars, C for trucks",
         "date": "YYYY-MM-DD"},
     "example": "/forecast?poste_id=1410&direction=1&vehicule=V&date=2025-03-12"},
    {"path": "/actuals/{poste_id}", "method": "GET",
     "description": "what the counter actually recorded, hour by hour, 2025 only",
     "example": "/actuals/1410"},
    {"path": "/docs", "method": "GET", "description": "interactive API documentation"},
]


@app.get('/')
def root():
    """Landing page -- tells a caller what this service offers."""
    return {"service": "Luxembourg Traffic Forecast", "version": "1.0",
            "trained_through": B['trained_through'],
            "endpoints": ENDPOINTS}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """On an unknown path, show the caller what they could have asked for.

    Starlette uses the literal detail 'Not Found' when no route matches. A 404
    raised by a handler (e.g. an unknown counter) carries its own message, so
    that is passed through untouched rather than buried under a route listing.
    """
    if exc.status_code == 404 and exc.detail == "Not Found":
        return JSONResponse(status_code=404, content={
            "error": f"Unknown path: {request.url.path}",
            "hint": "check the spelling, or use one of the endpoints below",
            "endpoints": ENDPOINTS})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get('/health')
def health():
    return {'status':'ok','trained_through':B['trained_through'],
            'expected_mae':B['expected_mae'],'series':len(B['meta'])}


@app.get("/counters")
def counters():
    m = B["meta"].sort_values("avg_per_hour", ascending=False)
    return {"count": len(m), "counters": [
        {"poste_id": int(r.POSTE_ID), "direction": int(r.DIRECTION),
         "vehicule": r.VEHICULE,
         "label": f"{r.route} — {r.localite} (dir {int(r.DIRECTION)}, "
                  f"{'cars' if r.VEHICULE == 'V' else 'trucks'})",
         "route": r.route, "localite": r.localite, "sens": r.sens,
         # LUREF (EPSG:2169) metres, exactly as recorded in the source CSV.
         # These are NOT lat/lon -- reproject to WGS84 before using on a map.
         "coord_x": float(r.coord_x), "coord_y": float(r.coord_y),
         # avg_per_hour covers days_reported days, NOT the full year -- no
         # counter reported all 366 days of 2024.
         "avg_per_hour": round(float(r.avg_per_hour), 1),
         "days_reported": int(r.days_reported),
         "first_day": r.first_day, "last_day": r.last_day} for r in m.itertuples()]}



@app.get("/actuals/{poste_id}")
def actuals(poste_id: int):
    """What this counter actually recorded, hour by hour, for every day of 2025.

    The counterpart to /forecast: that says what the model expects, this says
    what the road saw, and the difference is the only honest score of the model.

    Shape -- {"poste_id": 1410, "series": {"1-V": {"2025-02-15": [24 ints]}}},
    keyed "<direction>-<vehicule>" then by date.

    Sent straight off disk rather than parsed and re-serialised; the file is
    already the response body. A counter with no 2025 days has no file, and 404
    is the correct answer -- callers are expected to carry on without the line.

    poste_id is typed int, so no caller-supplied text reaches the path.
    """
    path = ACTUALS / f"{poste_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"No recorded 2025 data for counter {poste_id}")
    return FileResponse(path, media_type="application/json")


@app.get("/forecast")
def forecast(poste_id: int = Query(...), direction: int = Query(..., ge=1, le=2),
             vehicule: str = Query(..., pattern="^[VC]$"), date: str = Query(...)):
    try:
        out = forecast_dates(poste_id, direction, vehicule, date, bundle=B)
    except ValueError as ex:
        raise HTTPException(status_code=404, detail=str(ex))
    hol = bool(out["is_holiday"].any())
    return {"counter": {"poste_id": poste_id, "direction": direction,
                        "vehicule": vehicule},
            "date": date,
            "hourly": [{"hour": int(t.hour), "predicted": float(p),
                        "typical_2024": float(n)}
                       for t, p, n in zip(out["TIME_STAMP"], out["PREDICTED"],
                                          out["normal_for_slot"])],
            "daily_total": round(float(out["PREDICTED"].sum())),
            "is_holiday_period": hol,
            "expected_error": B["expected_mae_holiday"] if hol else B["expected_mae"]}