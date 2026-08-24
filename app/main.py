from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib

from app.forecast import forecast_dates

app = FastAPI(title='Luxembourg Traffic Forecast', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins='*', allow_methods=['GET'],
                   allow_headers=['*'])

B = joblib.load('models/forecast_model_2024.pkl')


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