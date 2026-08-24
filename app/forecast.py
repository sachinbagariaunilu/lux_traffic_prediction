import os
import logging
import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
GROUP = ['POSTE_ID', 'DIRECTION', 'VEHICULE']

# Path from the environment, with a container-friendly default.
# Loaded ONCE at import -- correct for a server. Never load per request.
MODEL_PATH = os.getenv('MODEL_PATH', 'models/forecast_model_2024.pkl')
B = joblib.load(MODEL_PATH)
log.info("loaded %s | %s | trained through %s",
         MODEL_PATH, B['kind'], B['trained_through'])


def forecast_dates(poste_id, direction, vehicule, start, end=None,
                   bundle=B, verbose=False):
    """Hourly traffic forecast for any date range. verbose=False in production."""
    a = pd.Timestamp(start).normalize()
    b = pd.Timestamp(end).normalize() if end else a
    ts = pd.date_range(a, b + pd.Timedelta(hours=23), freq='h')

    X = pd.DataFrame({'TIME_STAMP': ts})
    X['POSTE_ID'], X['DIRECTION'], X['VEHICULE'] = int(poste_id), int(direction), str(vehicule)
    X['HOUR']        = ts.hour
    X['DAY_OF_WEEK'] = ts.dayofweek
    X['IS_WEEKEND']  = (ts.dayofweek >= 5).astype('int8')
    X['HOUR_SIN']    = np.sin(2*np.pi*X['HOUR']/24.0)
    X['HOUR_COS']    = np.cos(2*np.pi*X['HOUR']/24.0)

    # holidays -- read from the bundle, so no notebook globals are needed
    dt = X['TIME_STAMP'].dt.normalize()
    X['IS_PUBLIC_HOLIDAY'] = dt.isin(bundle['public_holidays']).astype('int8')
    sh = np.zeros(len(X), dtype=bool)
    for s, e in bundle['school_ranges']:
        sh |= ((dt >= s) & (dt <= e)).to_numpy()
    X['IS_SCHOOL_HOLIDAY'] = sh.astype('int8')
    uniq = pd.Index(dt.unique())
    diff = np.abs(uniq.to_numpy()[:, None]
                  - bundle['public_holidays'].to_numpy()[None, :]) / np.timedelta64(1, 'D')
    X['DAYS_TO_HOLIDAY'] = dt.map(pd.Series(np.minimum(diff.min(axis=1), 7),
                                            index=uniq)).astype('float32')

    # profiles
    k = GROUP + ['DAY_OF_WEEK', 'HOUR']
    X['PROF_DOW_HOUR'] = X.merge(bundle['prof_dow_hour'], on=k, how='left')['PROF_DOW_HOUR'].values
    X['PROF_HOUR']     = X.merge(bundle['prof_hour'], on=GROUP+['HOUR'], how='left')['PROF_HOUR'].values
    t = X.merge(bundle['prof_series'], on=GROUP, how='left')
    X['PROF_MEAN'], X['PROF_STD'] = t['PROF_MEAN'].values, t['PROF_STD'].values

    if X['PROF_MEAN'].isna().all():
        raise ValueError(f"counter ({poste_id}, {direction}, '{vehicule}') is unknown "
                         f"to this model -- it was not in the training data")

    missing = [f for f in bundle['features'] if f not in X.columns]
    if missing:
        raise RuntimeError(f"features not built: {missing}")

    pred = np.clip(bundle['model'].predict(X[bundle['features']].astype('float32')), 0, None)

    out = pd.DataFrame({'TIME_STAMP': ts, 'PREDICTED': pred.round(1),
                        'normal_for_slot': X['PROF_DOW_HOUR'].round(1),
                        'is_public_holiday': X['IS_PUBLIC_HOLIDAY'],
                        'is_school_holiday': X['IS_SCHOOL_HOLIDAY']})
    out['is_holiday'] = ((out['is_public_holiday'] == 1) |
                         (out['is_school_holiday'] == 1)).astype('int8')

    if verbose:
        print(f"counter ({poste_id}, {direction}, '{vehicule}')  {a.date()} -> {b.date()}"
              f"  |  {len(out)} hours  |  daily total ~ {pred.sum()/((b-a).days+1):,.0f}")
        print(f"   expected error +/-{bundle['expected_mae']} veh/h")

    return out
