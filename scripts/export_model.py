
import joblib, pandas as pd

meta = (df_long.groupby(['POSTE_ID','DIRECTION','VEHICULE'], observed=True)
        .agg(route=('ROUTE','first'), localite=('LOCALITE','first'),
             sens=('SENS','first'), avg_per_hour=('TRAFFIC_VOLUME','mean'))
        .reset_index())
for c, t in [('POSTE_ID','int64'), ('DIRECTION','int64'), ('VEHICULE','str')]:
    meta[c] = meta[c].astype(t)

BUNDLE = {
    'model': PROD_MODEL,
    'features': FC_FEATURES,
    'prof_dow_hour': PROF_DH,
    'prof_hour': PROF_H,
    'prof_series': PROF_S,
    'meta': meta,
    'public_holidays': PUBLIC_HOLIDAYS,
    'school_ranges': SCHOOL_RANGES,
    'trained_through': str(all2024['TIME_STAMP'].max().date()),
    'expected_mae': 18.3,
    'expected_mae_holiday': 40.7,
    'expected_within_20pct': 77.8,
}
# compress=3 -- often halves the file, matters on a 512 MB tier
joblib.dump(BUNDLE, 'forecast_model_2024.pkl', compress=3)
print(f"saved. {len(meta):,} series available")
