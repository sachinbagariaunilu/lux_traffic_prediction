"""Add the missing 'meta' table to an existing model bundle.

/health and /counters both read bundle['meta'], which the shipped .pkl does not
have -- so both return HTTP 500. The bundle was written by a notebook cell that
predates the meta support in app/main.py.

This script does NOT retrain. It loads the existing bundle, keeps the trained
model and every other key exactly as-is, builds 'meta' from the 2024 source CSV,
and writes the bundle back out. Predictions are unchanged.

Usage:
    python scripts/patch_bundle_meta.py [--bundle PATH] [--csv PATH] [--dry-run]
"""
import argparse
import os
import shutil
import sys

import joblib
import pandas as pd

GROUP = ['POSTE_ID', 'DIRECTION', 'VEHICULE']
HOUR_COLS = [f'P{i:02d}_{i + 1:02d}' for i in range(24)]

# meta carries only what the source CSV actually contains. COORD_X / COORD_Y are
# LUREF (Luxembourg 1930 Gauss, EPSG:2169) in metres -- NOT latitude/longitude.
# A web map (Leaflet, Mapbox, Google) cannot use them directly; it needs WGS84.
# To add lat/lon later, reproject at build time so pyproj stays out of the
# serving image:
#     from pyproj import Transformer
#     t = Transformer.from_crs('EPSG:2169', 'EPSG:4326', always_xy=True)
#     meta['lon'], meta['lat'] = t.transform(meta['coord_x'], meta['coord_y'])
DEFAULT_CSV = os.getenv(
    'TRAFFIC_2024_CSV',
    '/Users/sachin.bagaria/Desktop/project/TEST/TEST/python-ecs-app/data/raw/'
    'donneestrafic-2024-DonneesTrafic_2024.csv')


def build_meta(csv_path, prof_series):
    """One row per (counter, direction, vehicle) that the model can actually serve."""
    need = GROUP + ['DATECOM', 'LOCALITE', 'ROUTE', 'SENS', 'COORD_X', 'COORD_Y'] + HOUR_COLS
    df = pd.read_csv(csv_path, usecols=need)
    df['DATECOM'] = pd.to_datetime(df['DATECOM'], format='%m/%d/%Y')

    # same filtering the model was trained with: 'U' is a redundant total,
    # direction 3 is a redundant combined direction
    df = df[(df['VEHICULE'] != 'U') & (~df['DIRECTION'].astype(str).isin(['3', '3.0']))]

    nan_cells = df[HOUR_COLS].isna().sum().sum()
    if nan_cells:
        print(f"  note: {nan_cells:,} NaN hour cells -- averaging ignores them")

    df['_avg'] = df[HOUR_COLS].mean(axis=1)
    meta = (df.groupby(GROUP, observed=True)
            .agg(route=('ROUTE', 'first'),
                 localite=('LOCALITE', 'first'),
                 sens=('SENS', 'first'),
                 coord_x=('COORD_X', 'first'),
                 coord_y=('COORD_Y', 'first'),
                 # sample size behind avg_per_hour. No counter reported all 366
                 # days of 2024, so this is NOT an annual average -- it covers
                 # only the days that counter actually reported.
                 days_reported=('DATECOM', 'nunique'),
                 first_day=('DATECOM', 'min'),
                 last_day=('DATECOM', 'max'),
                 avg_per_hour=('_avg', 'mean'))
            .reset_index())
    for c in ('first_day', 'last_day'):
        meta[c] = meta[c].dt.strftime('%Y-%m-%d')

    meta['POSTE_ID'] = meta['POSTE_ID'].astype('int64')
    meta['DIRECTION'] = meta['DIRECTION'].astype('int64')
    meta['VEHICULE'] = meta['VEHICULE'].astype('str')
    for c in ('route', 'localite', 'sens'):
        meta[c] = meta[c].fillna('').astype('str')

    # coordinates are recorded per site, so they must not vary within a POSTE_ID
    varying = df.groupby('POSTE_ID')[['COORD_X', 'COORD_Y']].nunique().gt(1).any(axis=1).sum()
    if varying:
        print(f"  WARNING: {varying} counters have more than one coordinate in the CSV; "
              f"kept the first")

    # Only advertise series the model can actually predict. prof_series is the
    # authoritative list -- forecast_dates() raises if a counter is missing there.
    before = len(meta)
    keys = prof_series[GROUP].copy()
    keys['POSTE_ID'] = keys['POSTE_ID'].astype('int64')
    keys['DIRECTION'] = keys['DIRECTION'].astype('int64')
    keys['VEHICULE'] = keys['VEHICULE'].astype('str')
    meta = meta.merge(keys, on=GROUP, how='inner')
    if before != len(meta):
        print(f"  dropped {before - len(meta)} series present in the CSV but not in the model")

    missing = len(keys) - len(meta)
    if missing:
        print(f"  WARNING: {missing} model series have no CSV metadata and will not be listed")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='models/forecast_model_2024.pkl')
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--dry-run', action='store_true', help='build and report, do not write')
    a = ap.parse_args()

    if not os.path.exists(a.bundle):
        sys.exit(f"bundle not found: {a.bundle}")
    if not os.path.exists(a.csv):
        sys.exit(f"2024 CSV not found: {a.csv}\nSet TRAFFIC_2024_CSV or pass --csv")

    print(f"loading {a.bundle} ...")
    B = joblib.load(a.bundle)
    print(f"  {B.get('kind', '?')}")
    print(f"  keys: {sorted(B)}")
    if 'meta' in B:
        print(f"  'meta' already present ({len(B['meta'])} rows) -- it will be rebuilt")

    print(f"building meta from {os.path.basename(a.csv)} ...")
    meta = build_meta(a.csv, B['prof_series'])
    print(f"  {len(meta)} series  |  model has {len(B['prof_series'])}")
    print(meta.sort_values('avg_per_hour', ascending=False).head(5).to_string(index=False))

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    B['meta'] = meta
    tmp = a.bundle + '.tmp'
    joblib.dump(B, tmp)

    # verify the rewritten file before replacing the original
    chk = joblib.load(tmp)
    assert 'meta' in chk and len(chk['meta']) == len(meta), "meta did not round-trip"
    assert chk['features'] == B['features'], "features changed"
    assert 'kind' in chk, "kind lost"

    backup = a.bundle + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(a.bundle, backup)
        print(f"\nbacked up original -> {backup}")
    os.replace(tmp, a.bundle)
    print(f"wrote {a.bundle}  ({os.path.getsize(a.bundle) / 1e6:.1f} MB)")
    print(f"verified: {len(chk['meta'])} series, {len(chk['features'])} features, kind intact")


if __name__ == '__main__':
    main()
