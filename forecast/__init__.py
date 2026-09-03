"""Luxembourg hourly traffic forecasting.

Predicts TRAFFIC_VOLUME (vehicles/hour) for a series -- one
(POSTE_ID, DIRECTION, VEHICULE) combination. 1,058 series across 270 sites.

The shipped model is LightGBM with 12 features and NO lag features. That is the
central design decision: it never reads recent traffic, so it can predict any
date arbitrarily far ahead with no live data feed. See IMPROVEMENTS.md 1a.

Module map -- import order is also the dependency order:

    config        paths, TRAIN_YEARS, GROUP/TARGET, the leakage guard
    calendar_lux  public + school holidays, holiday features, coverage guard
    data          raw CSV -> long frame (melt, drop aggregates, parquet cache)
    features      clock features, profile tables, feature assembly
    train         fit, bundle, save
    evaluate      score() and the diagnostic breakdowns
    predict       load a bundle and forecast arbitrary dates

Nothing is imported eagerly here: `import forecast` must stay cheap, because
config.assert_no_leakage() is meant to run before any data is touched.

Usage:
    from forecast import config, data, features
    config.assert_no_leakage()
    df = data.load_years(config.TRAIN_YEARS)
"""
from __future__ import annotations

__all__ = [
    "config",
    "calendar_lux",
    "data",
    "features",
    "train",
    "evaluate",
    "predict",
]
