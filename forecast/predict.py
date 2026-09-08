"""Load a bundle and forecast arbitrary dates.

This is the product: one .pkl that answers any date with no database, no
ingestion pipeline, no freshness monitoring. Nothing here reads a sensor value.

    from forecast import predict
    b = predict.load_bundle()
    predict.forecast_dates(26, 1, "V", "2028-12-25", bundle=b)

The model never sees a date. By the time it is called, the date has become
twelve numbers -- clock features computed by arithmetic, calendar features by
LOOKUP against data/external/*.json, profile features by lookup against tables
frozen in the bundle. That lookup is why a 2028 prediction works at all, and it
is exactly what was broken in the legacy model: its calendar ended 2025-12-26,
so every date after that got IS_PUBLIC_HOLIDAY=0 and a confident wrong answer.

Measured on the current bundle, counter 26 cars at 08:00:

    2028-12-25 (Christmas Monday)   143.5 veh/h
    2028-11-27 (ordinary Monday)   1146.4 veh/h

The legacy model said 1155 for Christmas.

NOTE the holiday lists are NOT baked into the .pkl -- they are read from
data/external/ at call time. Update the JSON when MENJE publishes 2030 and every
existing bundle handles 2030 with no retraining. The cost is that the .pkl alone
is not self-contained: deploying it means shipping data/external/ too.

Source: legacy notebook cell 85.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import calendar_lux as cal
from . import config, features


def load_bundle(path: Path | None = None) -> dict:
    """Load a bundle and validate it before returning.

    Refuses a bundle that cannot be used safely: no feature list (order
    matters), no profiles, or no calendar horizon to enforce the guard against.
    """
    # Coerce: a str is the natural thing to pass from a server config or an
    # env var, and .exists() on one raises AttributeError several lines later.
    path = Path(path) if path is not None else (config.MODELS / "forecast_model.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"no bundle at {path}. Run:  python -m forecast.train")

    bundle = joblib.load(path)
    for key in ("model", "features", "profiles", "calendar_through"):
        if key not in bundle:
            raise ValueError(f"bundle at {path} is missing {key!r}")
    if not bundle["features"]:
        raise ValueError("bundle has an empty feature list")
    return bundle


def _build_features(
    series: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    bundle: dict,
) -> pd.DataFrame:
    """The six steps, in order. Shared by forecast_dates and forecast_series.

    One definition, so the single-series and many-series paths cannot drift --
    which is precisely how scripts/diagnose.py silently reported 14.04 against
    train.py's 13.95 for an hour.
    """
    X = series.merge(pd.DataFrame({"TIME_STAMP": timestamps}), how="cross")

    # month=True since 2026-09-08: MONTH_SIN/COS are in the shipped feature set
    # (features.FC_FEATURES_WITH_MONTH). Building them unconditionally costs two
    # float32 columns and keeps this path able to serve BOTH bundle vintages --
    # build_matrix selects by name, so a 14-feature bundle simply ignores them.
    # Dropping the flag would make build_matrix raise KeyError on a 16-feature
    # bundle, which is the loud failure we want rather than a silent one.
    X = features.add_clock_features(X, month=True)        # arithmetic
    X = cal.add_holiday_features(X)                      # lookup: JSON
    if bundle["profiles"].term_split:
        X = cal.add_term_split_profiles_key(X)
    X = features.attach_profiles(X, bundle["profiles"], warn_missing=False)
    return X


def forecast_dates(
    poste_id: int,
    direction: int,
    vehicule: str,
    start,
    end=None,
    *,
    bundle: dict | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Hourly forecast for one series over [start, end], inclusive.

    Raises rather than guessing in two cases:
      - the range extends past the calendar horizon (would be holiday-blind)
      - the series is unknown to the bundle's profiles (no history to stand on)

    Prints the provenance caveat for thin or truncated series unless quiet=True.
    Those warnings are honesty, not error correction -- a truncated profile
    measured within 0.04 MAE of a full one.
    """
    bundle = bundle or load_bundle()
    a = pd.Timestamp(start).normalize()
    b = pd.Timestamp(end).normalize() if end is not None else a

    cal.assert_covers(a, b)                              # GUARD FIRST

    series = pd.DataFrame({"POSTE_ID": [np.int32(poste_id)],
                           "DIRECTION": [np.int8(direction)],
                           "VEHICULE": [str(vehicule)]})
    ts = pd.date_range(a, b + pd.Timedelta(hours=23), freq="h")
    X = _build_features(series, ts, bundle)

    if X["PROF_MEAN"].isna().all():
        raise ValueError(
            f"counter ({poste_id}, {direction}, {vehicule!r}) is unknown to this "
            f"model -- it has no history in {bundle['profiles'].built_from}")

    unmatched = int(X["PROF_DOW_HOUR"].isna().sum())
    pred = np.clip(bundle["model"].predict(
        features.build_matrix(X, bundle["features"])), 0, None)

    if not quiet:
        note = bundle["profiles"].series_note(poste_id, direction, vehicule)
        if note:
            print(f"  note: {note}")
        if unmatched:
            print(f"  note: {unmatched} of {len(X)} hours had no (weekday, hour) "
                  f"profile; those predictions are weaker than they look")

    return pd.DataFrame({
        "TIME_STAMP": X["TIME_STAMP"].to_numpy(),
        "PREDICTED": pred.round(1),
        "typical_for_slot": X["PROF_DOW_HOUR"].round(1).to_numpy(),
        "is_public_holiday": X["IS_PUBLIC_HOLIDAY"].to_numpy(),
        "is_school_holiday": X["IS_SCHOOL_HOLIDAY"].to_numpy(),
    })


def forecast_series(
    series: pd.DataFrame,
    start,
    end=None,
    *,
    bundle: dict | None = None,
) -> pd.DataFrame:
    """forecast_dates for many series at once, in ONE vectorised predict call.

    The notebook only ever predicted a single series interactively. Anything
    real -- a network daily total, a monthly report -- needs all 1,058, and
    1,058 separate predict() calls is not the way.

    `series` needs POSTE_ID, DIRECTION, VEHICULE. Pass
    bundle["profiles"].provenance[config.GROUP] for the full set.
    """
    bundle = bundle or load_bundle()
    a = pd.Timestamp(start).normalize()
    b = pd.Timestamp(end).normalize() if end is not None else a

    cal.assert_covers(a, b)

    ts = pd.date_range(a, b + pd.Timedelta(hours=23), freq="h")
    X = _build_features(features._plain_keys(series[config.GROUP].copy()), ts, bundle)

    unknown = X.loc[X["PROF_MEAN"].isna(), config.GROUP].drop_duplicates()
    if len(unknown):
        print(f"  note: {len(unknown)} series unknown to this model, dropped")
        X = X[X["PROF_MEAN"].notna()]

    X["PREDICTED"] = np.clip(bundle["model"].predict(
        features.build_matrix(X, bundle["features"])), 0, None).round(1)
    return X[[*config.GROUP, "TIME_STAMP", "PREDICTED",
              "IS_PUBLIC_HOLIDAY", "IS_SCHOOL_HOLIDAY"]]


def apply_level_index(pred: np.ndarray, target_year: int, index: dict) -> np.ndarray:
    """Scale predictions by a projected annual level index. WILL NOT BE BUILT.

    Closed 2026-09-02, on evidence, not for lack of data. The model assumes ZERO
    growth and that is now a decision rather than a gap.

    An earlier version of this docstring quoted +0.70%/yr overall and +1.83% for
    freight, projecting ~2.8% and ~7.5% by 2028. Those came from UNPAIRED annual
    means, which coverage differences distort: a counter reporting 14% of one
    year and 98% of the next shows growth that is really just more observations.
    Recomputed on paired (counter, month, weekday, hour) slots by
    scripts/measure_growth.py:

        period        network   freight(C)   cars(V)   per-counter std
        2023->2024     +0.13%     -0.72%      +0.18%        9.67
        2024->2025     +1.03%     +0.53%      +1.06%        8.40

    Network direction is consistent (both positive) but the magnitude spans
    7.8x, freight changes SIGN, and per-counter noise is ~9pp against a sub-1pp
    signal. Published PCH counts agree: four counters spanning -0.8% to
    +4.8%/yr, one of them shrinking. There is no defensible number to project.

    The hourly stake is negligible anyway -- 0.5%/yr is ~0.6 veh/h by 2028
    against an MAE of 14, some 25x smaller than the random error. Where a level
    gap DOES matter is multi-year annual TOTALS, because random error cancels
    over thousands of counter-hours and systematic bias never does. That case is
    served by measure_growth.reporting_factor(year), applied by the caller and
    quoted with its band (+2.0% central for 2028, 0-4% range) so the assumption
    stays visible instead of being buried in the .pkl.

    Do NOT add YEAR as a model feature either. LightGBM splits on thresholds, so
    a 2028 row lands in the same leaf as the latest year seen in training and the
    model predicts that year's level for 2028, 2035 and 3000 alike -- zero trend
    extrapolation even with three years showing clear growth.
    """
    raise NotImplementedError(
        "closed on evidence -- growth is not measurable at this horizon. "
        "See docs/TODO_IMPROVEMENTS.md 'P1 CLOSED' and "
        "scripts/measure_growth.py's growth policy.")
