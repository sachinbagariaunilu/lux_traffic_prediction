
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# The 12 features of the SHIPPED model, in bundle order. Order matters: the
# model is fit on this sequence and predict must reproduce it exactly.
FC_FEATURES: list[str] = [
    "HOUR", "HOUR_SIN", "HOUR_COS", "DAY_OF_WEEK", "IS_WEEKEND",
    "PROF_DOW_HOUR", "PROF_HOUR", "PROF_MEAN", "PROF_STD",
    "IS_PUBLIC_HOLIDAY", "IS_SCHOOL_HOLIDAY", "DAYS_TO_HOLIDAY",
]

# ADOPTED 2026-09-08. Annual seasonality -- the largest single feature gain in
# this project's history, and the last one available.
#
# Built in notebook cell 45 but DROPPED from FC_FEATURES. The stated reason --
# "training stops in September, October is a month the trees have never seen" --
# was correct for the dev split and expired once the production model trained on
# all 12 months. It then stayed off for years because the 46-day Nov-Dec split
# kept SCORING it as harmful, which it cannot help doing: that split trains
# Jan-Sep, so November and December are extrapolation and August is absent from
# both sides. Measured on the two windows:
#
#     46-day blind test (Nov-Dec)     14.01 -> 15.76   +1.75   <- the artefact
#     full year, train 2024/score 2025 13.79 -> 13.02  -0.77   <- the truth
#
# Same feature, opposite verdict. The full year is the honest one, and it is the
# only window in which a seasonal feature CAN be judged.
#
# Where the -0.77 comes from, by month (train 2024 -> score 2025):
#     Aug -2.19   Dec -2.17   Jan -1.18   May -1.16   Jul -0.84   Apr -0.75
#     Jun/Oct/Nov -0.35   Feb/Sep -0.23   Mar +0.73  <- only regression
# August and December were the two worst months for the 14-feature model and
# are the two the old split could never measure. March regressed because Easter
# moved (31 Mar 2024 -> 20 Apr 2025) and a smooth month encoding cannot
# represent a moving feast; with one training year it learned "late March is
# depressed" from a single Easter.
#
# Improves every regime, largest gains on the weekend:
#     Sunday      13.8% of its FIXABLE error removed
#     Saturday    11.2%
#     all rows     9.3%
#     workday peak 6.9%
MONTH_FEATURES: list[str] = ["MONTH_SIN", "MONTH_COS"]

LAG_FEATURES: list[str] = [
    "LAG_1", "LAG_2", "LAG_24", "LAG_168",
    "ROLLING_MEAN_3H", "ROLLING_STD_3H", "ROLLING_MEAN_24H",
]

# FINDING #5. These survive the melt and are used in ZERO model features. The
# model sees each series only through four profile numbers -- it never learns
# that a counter is on a motorway, in a city, or next to another counter.
# Phase 4 candidate via add_counter_identity().
UNUSED_RAW_COLUMNS: list[str] = [
    "LOCALITE", "ROUTE", "SENS", "D1", "D2", "COORD_X", "COORD_Y",
]

# Phase 4 item 4. PROF_STD is currently the model's only signal about spread;
# percentiles tell it "this slot is usually 300 but sometimes 600", which is
# real information about how spiky a slot is.
ROBUST_PROFILE_FEATURES: list[str] = ["PROF_MEDIAN", "PROF_P90", "PROF_IQR"]

# FINDING #17. VEHICULE is absent from FC_FEATURES, so the model cannot tell a
# lorry series from a car series except indirectly through PROF_MEAN. Measured
# on 2024 they are two different distributions sharing one set of trees:
#
#   C  4,449,816 rows (50%)  mean  10.42  zeros 18.7%  skew 5.98  -> 26% rel err
#   V  4,449,816 rows (50%)  mean 183.33  zeros  1.5%  skew 3.37  -> 14% rel err
#
# 17x the scale, 12x the zero rate. Add via add_vehicle_class(); OFF by default
# so its effect can be measured on its own.
VEHICLE_FEATURES: list[str] = ["IS_HEAVY"]

FC_FEATURES_WITH_CLASS: list[str] = [*FC_FEATURES, *VEHICLE_FEATURES]

# Public-holiday profiles. The model already HAS an IS_PUBLIC_HOLIDAY flag, but
# PROF_DOW_HOUR is not keyed on it -- so on Christmas Day the profile still hands
# over the average of every Wednesday 08:00 at that counter, Christmas included.
# You cannot un-blend an average, which is the same argument that justified the
# school-term split in calendar_lux.add_term_split_profiles_key().
#
# Measured, profile used as a DIRECT predictor on 2024:
#     blended                              14.71
#     + school term split                  13.23   <- implemented (TERM_SPLIT)
#     + public holiday split               11.73   <- this
#
# Adding IS_PUBLIC_HOLIDAY to the dow_hour key does NOT work: with 10 holiday
# days a (series, dow, hour, holiday) cell holds a MEDIAN OF 2 OBSERVATIONS and
# 100% of cells have under 5. Measured on 2024:
#
#     (series, dow, hour, holiday)  124,848 cells   median  2 obs   100% under 5
#     (series, hour, holiday)        25,392 cells   median 10 obs     1% under 5
#     (series, holiday)               1,058 cells   median 240 obs    0% under 5
#
# So DAY_OF_WEEK is dropped for the holiday case -- a public holiday behaves like
# a Sunday whichever weekday it lands on -- giving two usable shapes:
#
#   PROF_HOLIDAY_HOUR   this counter's mean at this HOUR across holiday rows
#   PROF_HOLIDAY_RATIO  this counter's holiday mean / overall mean, one scalar
#
# Both are defined on EVERY row, not just holiday rows: they describe the
# counter, and the model combines them with IS_PUBLIC_HOLIDAY itself.
#
# OFF by default so the effect is measured alone.
HOLIDAY_PROFILE_FEATURES: list[str] = ["PROF_HOLIDAY_HOUR", "PROF_HOLIDAY_RATIO"]

FC_FEATURES_WITH_HOLIDAY: list[str] = [*FC_FEATURES, *HOLIDAY_PROFILE_FEATURES]

# THE SHIPPED SET as of 2026-09-08. See MONTH_FEATURES for the measurement.
#
# Anything appended here must also be buildable by predict.py, which reassembles
# every feature from a bare date. add_clock_features(month=True) is what supplies
# the two month columns, and predict.py calls it -- if that call ever loses its
# month=True, build_matrix raises KeyError rather than predicting nonsense.
FC_FEATURES_WITH_MONTH: list[str] = [*FC_FEATURES_WITH_HOLIDAY, *MONTH_FEATURES]

# C = Camions (lorries), V = Vehicules legers (cars). Inferred from the volumes
# -- C averages 10.4 veh/h against V's 183.3, and heavy traffic is the rarer of
# the two. The notebook's own plot labels contradict each other on this: cell 25
# says "C = Heavy/Trucks, V = Light/Cars" while cell 31 says "C = Cars, V =
# Heavy Vehicles". Cell 25 agrees with the data and with cell 26's text.
HEAVY_VEHICLE_CODE = "C"

PROFILE_PREFIX = "PROF_"
_DOW_HOUR_KEYS = [*config.GROUP, "DAY_OF_WEEK", "HOUR"]
_HOUR_KEYS = [*config.GROUP, "HOUR"]


@dataclass(frozen=True)
class Profiles:
    """The frozen profile tables that ship inside the bundle.

    dow_hour  mean per (POSTE_ID, DIRECTION, VEHICULE, DAY_OF_WEEK, HOUR)
    hour      mean per (POSTE_ID, DIRECTION, VEHICULE, HOUR)
    series    mean + std per (POSTE_ID, DIRECTION, VEHICULE)
    provenance  per series: N_OBS, FIRST_SEEN, LAST_SEEN, COVERAGE. NOT merged
                onto the training frame -- it exists so predict can tell a user
                what the profile was built from. See series_note().
    built_from  e.g. "2024-01-01..2024-09-30" -- so a bundle can always answer
                which rows its profiles saw, which is what decides whether a
                given evaluation is honest.
    term_split  True if dow_hour is additionally keyed on IS_SCHOOL_TERM.
    """

    dow_hour: pd.DataFrame
    hour: pd.DataFrame
    series: pd.DataFrame
    provenance: pd.DataFrame
    built_from: str
    term_split: bool = False
    holiday_hour: pd.DataFrame | None = None
    holiday_split: bool = False

    @property
    def dow_hour_keys(self) -> list[str]:
        return [*_DOW_HOUR_KEYS, "IS_SCHOOL_TERM"] if self.term_split else _DOW_HOUR_KEYS

    def series_note(
        self,
        poste_id: int,
        direction: int,
        vehicule: str,
        *,
        coverage_floor: float = 0.60,
        stale_days: int = 30,
    ) -> str | None:
        """Human-readable caveat for one series, or None if it looks healthy.

        THRESHOLDS ARE DELIBERATELY LOW. Two different conditions get confused
        here, and only one is harmful:

          low coverage, full span   scattered missing hours -- MEASURED HARMLESS.
                                    Every coverage tier scored 19-23% error, so
                                    a counter at 89% is fine.
          truncated                 stopped reporting, so the profile never saw
                                    later seasons -- genuinely worth flagging.

        coverage_floor was 0.90, which fired on 94 of 1,058 series, almost all
        of them healthy counters with scattered gaps. At 0.60 it fires on 12,
        and the truncation check catches the 16 series (4 counters) that
        actually stopped. A warning that cries wolf 94 times is a warning
        nobody reads.

        Honesty, not error correction. Measured on 2024, a profile truncated to
        January-April predicted November-December within 0.04 MAE of a full
        January-September one -- so a thin profile is not a measurable accuracy
        problem, and neighbour-borrowing would patch nothing.

        But someone asking for a 2027 forecast on a counter that last reported
        in April 2024 should be told, and nothing else in the pipeline tells
        them. Note the test window was cool and term-time; a summer forecast
        from a winter-only profile is less well evidenced.
        """
        p = self.provenance
        row = p[(p.POSTE_ID == int(poste_id))
                & (p.DIRECTION == int(direction))
                & (p.VEHICULE == str(vehicule))]
        if row.empty:
            return (f"counter ({poste_id}, {direction}, {vehicule!r}) is unknown to "
                    f"these profiles (built from {self.built_from})")

        r = row.iloc[0]
        notes = []
        if r.COVERAGE < coverage_floor:
            notes.append(f"reported only {100 * r.COVERAGE:.0f}% of the training window")
        trained_to = pd.Timestamp(self.built_from.split("..")[1])
        gap = (trained_to - pd.Timestamp(r.LAST_SEEN).normalize()).days
        if gap > stale_days:
            notes.append(f"last seen {pd.Timestamp(r.LAST_SEEN):%Y-%m-%d}, "
                         f"{gap} days before the end of training")
        if not notes:
            return None
        return (f"counter ({poste_id}, {direction}, {vehicule!r}): "
                + "; ".join(notes)
                + f". Its profile is built from {int(r.N_OBS):,} hours and may not "
                  f"represent seasons it never observed.")


# --------------------------------------------------------------------------
# Clock features
# --------------------------------------------------------------------------


def add_clock_features(
    d: pd.DataFrame,
    *,
    ts_col: str = "TIME_STAMP",
    month: bool = False,
) -> pd.DataFrame:
    """Attach HOUR, DAY_OF_WEEK, IS_WEEKEND, HOUR_SIN, HOUR_COS.

    Sin/cos encode the 24h wrap so hour 23 sits next to hour 0 rather than 23
    units away. Without them the model would think midnight is maximally far
    from 23:00, when it is one hour later.

    Everything here is derived from the timestamp alone, which is why it works
    for 2028 as readily as for 2024.

    month=True also adds MONTH_SIN/COS -- off by default; see MONTH_FEATURES.

    Dtypes are pinned. The notebook flipped IS_WEEKEND between bool and int
    across cells 34 and 48, and a dtype that differs between train and predict
    is a silent accuracy leak.
    """
    d = d.copy()
    ts = d[ts_col]

    d["HOUR"] = ts.dt.hour.astype("int8")
    d["DAY_OF_WEEK"] = ts.dt.dayofweek.astype("int8")
    d["IS_WEEKEND"] = (ts.dt.dayofweek >= 5).astype("int8")

    radians = 2 * np.pi * d["HOUR"].to_numpy() / 24.0
    d["HOUR_SIN"] = np.sin(radians).astype("float32")
    d["HOUR_COS"] = np.cos(radians).astype("float32")

    if month:
        m = 2 * np.pi * ts.dt.month.to_numpy() / 12.0
        d["MONTH_SIN"] = np.sin(m).astype("float32")
        d["MONTH_COS"] = np.cos(m).astype("float32")

    return d


def add_vehicle_class(d: pd.DataFrame) -> pd.DataFrame:
    """Attach IS_HEAVY -- 1 for lorries (C), 0 for cars (V). FINDING #17.

    Use FC_FEATURES_WITH_CLASS to actually feed it to the model. Off the default
    feature list so its effect gets measured alone.

    Why a 0/1 column rather than a LightGBM categorical: VEHICULE has exactly
    two levels after drop_aggregates, and for a tree those are identical. A
    split on `IS_HEAVY <= 0.5` separates the classes perfectly -- categorical
    handling only earns its cost at three or more levels, where LightGBM can
    group levels into arbitrary subsets. This way also needs no
    `categorical_feature=` plumbing in fit(), and no dtype exception in
    build_matrix(), which casts everything to float32.

    If a third class ever appears, switch to a real categorical -- the assert
    below will fire and tell you.
    """
    d = d.copy()
    levels = set(d["VEHICULE"].astype(str).unique())
    if not levels <= {"C", "V"}:
        raise ValueError(
            f"expected only C and V after drop_aggregates, found {sorted(levels)}. "
            f"With 3+ levels a binary column is no longer equivalent to a "
            f"categorical -- use a LightGBM categorical_feature instead.")
    d["IS_HEAVY"] = (d["VEHICULE"].astype(str) == HEAVY_VEHICLE_CODE).astype("int8")
    return d


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def build_profiles(
    train: pd.DataFrame,
    *,
    split_by_term: bool = False,
    robust: bool = False,
    holiday_split: bool = False,
) -> Profiles:
    """Group-by averages over TRAINING rows only.

    These are target encoding: each value is an average of the thing being
    predicted. Build them from rows you will NOT score on, or the score is
    flattered. See IMPROVEMENTS.md Section 3.

    split_by_term=True additionally keys dow_hour on IS_SCHOOL_TERM, so a
        "Tuesday 08:00" average stops blending busy term-time Tuesdays with
        quiet August ones (finding #6). Measured as a direct predictor on 2024:
        14.71 blended -> 13.23 split, 10.1% better in-sample. The gain to the
        MODEL is smaller, since it already has IS_SCHOOL_HOLIDAY -- measure it.
        Requires calendar_lux.add_term_split_profiles_key() first.
    robust=True adds ROBUST_PROFILE_FEATURES to the series table. Phase 4.
    """
    if split_by_term and "IS_SCHOOL_TERM" not in train.columns:
        raise ValueError(
            "split_by_term=True needs IS_SCHOOL_TERM; call "
            "calendar_lux.add_term_split_profiles_key() first")

    target = config.TARGET
    dow_keys = [*_DOW_HOUR_KEYS, "IS_SCHOOL_TERM"] if split_by_term else _DOW_HOUR_KEYS

    dow_hour = (train.groupby(dow_keys, observed=True)[target]
                .mean().rename("PROF_DOW_HOUR").reset_index())
    hour = (train.groupby(_HOUR_KEYS, observed=True)[target]
            .mean().rename("PROF_HOUR").reset_index())

    agg: dict[str, tuple[str, object]] = {
        "PROF_MEAN": (target, "mean"),
        "PROF_STD": (target, "std"),
    }
    if robust:
        agg["PROF_MEDIAN"] = (target, "median")
        agg["PROF_P90"] = (target, lambda s: s.quantile(0.90))
        agg["PROF_IQR"] = (target, lambda s: s.quantile(0.75) - s.quantile(0.25))
    series = train.groupby(config.GROUP, observed=True).agg(**agg).reset_index()

    holiday_hour = None
    if holiday_split:
        if "IS_PUBLIC_HOLIDAY" not in train.columns:
            raise ValueError(
                "holiday_split=True needs IS_PUBLIC_HOLIDAY; call "
                "calendar_lux.add_holiday_features() first")
        hol = train[train["IS_PUBLIC_HOLIDAY"] == 1]
        if hol.empty:
            raise ValueError(
                "no public-holiday rows in the training window, so a holiday "
                "profile cannot be built. Check the calendar covers these dates.")

        # Keyed on (series, HOUR) -- NOT DAY_OF_WEEK. See HOLIDAY_PROFILE_FEATURES:
        # adding dow leaves a median of 2 observations per cell.
        holiday_hour = (hol.groupby(_HOUR_KEYS, observed=True)[target]
                        .mean().rename("PROF_HOLIDAY_HOUR").reset_index())

        # One scalar per series: how much this counter changes on a holiday.
        # Ratio rather than a level, so it stays meaningful across the 680x
        # volume range between a motorway and a village lane.
        hol_mean = hol.groupby(config.GROUP, observed=True)[target].mean()
        series = series.merge(
            (hol_mean / series.set_index(config.GROUP)["PROF_MEAN"])
            .rename("PROF_HOLIDAY_RATIO").reset_index(),
            on=config.GROUP, how="left")
        # 1.0 = "no holiday effect known" -- a safe, neutral default for a
        # series that reported no holiday hours. Counted, not swallowed.
        missing = int(series["PROF_HOLIDAY_RATIO"].isna().sum())
        if missing:
            print(f"note: {missing} of {len(series)} series have no holiday "
                  f"observations; PROF_HOLIDAY_RATIO defaults to 1.0 for them")
        series["PROF_HOLIDAY_RATIO"] = series["PROF_HOLIDAY_RATIO"].fillna(1.0)

    span = train["TIME_STAMP"]
    span_hours = (span.max() - span.min()) / pd.Timedelta(hours=1) + 1
    provenance = (train.groupby(config.GROUP, observed=True)["TIME_STAMP"]
                  .agg(N_OBS="size", FIRST_SEEN="min", LAST_SEEN="max")
                  .reset_index())
    provenance["COVERAGE"] = provenance["N_OBS"] / span_hours

    return Profiles(
        dow_hour=_plain_keys(dow_hour),
        hour=_plain_keys(hour),
        series=_plain_keys(series),
        provenance=_plain_keys(provenance),
        built_from=f"{span.min():%Y-%m-%d}..{span.max():%Y-%m-%d}",
        term_split=split_by_term,
        holiday_hour=_plain_keys(holiday_hour) if holiday_hour is not None else None,
        holiday_split=holiday_split,
    )


def _plain_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Pin the merge-key dtypes to match the long frame produced by data.py.

    The notebook carried a _plain() helper whose comment claimed "category dtype
    makes lookups return NaN silently". Tested on pandas 2.3.3, that is NOT true:
    int32/int64, category/object, float/int and differing category sets all merge
    correctly, and only int-vs-str fails -- loudly, with a ValueError.

    So this is NOT protection against silent NaN merges. What it actually buys:

      - consistency. data.drop_aggregates() emits POSTE_ID int32 / DIRECTION
        int8 / VEHICULE str; groupby may hand back int64. Pinning both sides
        keeps the frame's dtypes stable through a merge.
      - memory. int32 over int64 halves the key columns in tables that ship
        inside the bundle and get merged on every prediction.
      - determinism. Merge dtype coercion has changed across pandas versions and
        may again. Fixing the types means our behaviour does not depend on that.
      - it repairs int-vs-str, e.g. a POSTE_ID column read from CSV as text.

    The genuine silent-NaN trap is elsewhere and we already avoid it:
    groupby(observed=False) on a categorical emits a row for every UNOBSERVED
    level, whose mean is NaN, and that NaN merges cleanly onto real rows.
    build_profiles() uses observed=True throughout.
    """
    table = table.copy()
    for col, dtype in (("POSTE_ID", "int32"), ("DIRECTION", "int8")):
        if col in table.columns:
            table[col] = table[col].astype(dtype)
    if "VEHICULE" in table.columns:
        table["VEHICULE"] = table["VEHICULE"].astype(str)
    return table


def attach_profiles(
    d: pd.DataFrame,
    profiles: Profiles,
    *,
    warn_missing: bool = True,
) -> pd.DataFrame:
    """Left-merge the profile columns onto `d`.

    Idempotent: existing PROF_* columns are dropped first. Without that, a
    second call collides and pandas renames both sides to _x/_y until the _x
    name itself collides and raises MergeError.

    Reports NaN counts rather than swallowing them. PROF_MEAN is NaN only for a
    series absent from training, but PROF_DOW_HOUR can be NaN for individual
    (weekday, hour) slots of a KNOWN series -- and LightGBM accepts NaN natively
    and returns a plausible number with no warning (finding #13).
    """
    out = d.drop(columns=[c for c in d.columns if c.startswith(PROFILE_PREFIX)])
    before = len(out)

    tables = [(profiles.dow_hour, profiles.dow_hour_keys),
              (profiles.hour, _HOUR_KEYS),
              (profiles.series, config.GROUP)]
    if profiles.holiday_hour is not None:
        tables.append((profiles.holiday_hour, _HOUR_KEYS))

    for table, keys in tables:
        missing_keys = [k for k in keys if k not in out.columns]
        if missing_keys:
            raise KeyError(f"cannot merge profiles: {out.columns.name or 'frame'} "
                           f"is missing join keys {missing_keys}")
        out = out.merge(table, on=keys, how="left")

    # A counter absent from the holiday table has no measured holiday behaviour.
    # Fall back to its ordinary hourly profile, which says "expect a normal
    # hour" -- neutral, and better than a NaN LightGBM would silently absorb.
    if "PROF_HOLIDAY_HOUR" in out.columns:
        gap = out["PROF_HOLIDAY_HOUR"].isna()
        if gap.any():
            out.loc[gap, "PROF_HOLIDAY_HOUR"] = out.loc[gap, "PROF_HOUR"]

    if len(out) != before:
        raise AssertionError(
            f"profile merge changed the row count {before:,} -> {len(out):,}; a "
            f"profile table has duplicate keys")

    if warn_missing:
        prof_cols = [c for c in out.columns if c.startswith(PROFILE_PREFIX)]
        nulls = out[prof_cols].isna().sum()
        if nulls.any():
            share = {c: f"{100 * n / len(out):.2f}%" for c, n in nulls.items() if n}
            print(f"note: unmatched profile values {share} -- LightGBM accepts "
                  f"NaN silently, so these predictions are weaker than they look")

    return out


def add_counter_identity(
    d: pd.DataFrame,
    attributes: pd.DataFrame,
    *,
    categorical: bool = True,
    coords: bool = True,
    neighbours: int = 0,
) -> pd.DataFrame:
    """Expose the counter's identity and location to the model. FINDING #5.

    NOT IMPLEMENTED -- Phase 4. Recorded here so the design decision is not lost.

    UNUSED_RAW_COLUMNS carry real information the model has never seen. Three
    increasingly ambitious options:

    categorical=True  POSTE_ID / ROUTE / LOCALITE as LightGBM native
        categoricals, so the model can learn per-counter and per-road behaviour
        beyond the four profile numbers.
    coords=True       COORD_X / COORD_Y. Cell 29 measured a near-zero LINEAR
        correlation with volume and concluded coordinates were useless -- that
        conclusion does not follow. Trees split on thresholds and need no
        linearity, so the correlation says nothing about tree usefulness.
    neighbours>0      mean profile of the N nearest counters, letting a series
        borrow signal from its corridor. The only option that could help a
        counter absent from training, which predict.forecast_dates() currently
        rejects outright.

    High cardinality (270 sites, 107 routes) risks overfitting, so each option
    needs its own held-out measurement.
    """
    raise NotImplementedError("Phase 4 -- see IMPROVEMENTS.md Section 9")


# --------------------------------------------------------------------------
# Lags -- nowcast path only
# --------------------------------------------------------------------------


def add_lag_features(d: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    """Attach LAG_FEATURES. NOWCAST ONLY -- never on the forecast path.

    Every lag is GROUPED by series and sorted by time within the group.
    Ungrouped shift/rolling reaches across series boundaries and pulls a
    motorway's values into a village lane -- the bug that voided the notebook's
    MA and AR baselines (findings #10, #11). AutoReg went further and fitted
    1,058 concatenated series as one univariate series, which makes those
    results void rather than merely weak.

    dropna=True removes the 168h warm-up rows. The caller's choice, deliberately
    -- doing it unconditionally inside data.py is bug 16, which cost 177,744
    rows and New Year's Day for a feature the shipped model never uses.
    """
    out = d.sort_values([*config.GROUP, "TIME_STAMP"]).reset_index(drop=True)
    grouped = out.groupby(config.GROUP, observed=True)[config.TARGET]

    for lag in (1, 2, 24, 168):
        out[f"LAG_{lag}"] = grouped.shift(lag).astype("float32")

    # shift(1) first: a rolling window including the current hour would leak the
    # answer into its own feature.
    shifted = grouped.shift(1)
    by_series = shifted.groupby([out[c] for c in config.GROUP], observed=True)
    out["ROLLING_MEAN_3H"] = by_series.transform(
        lambda s: s.rolling(3, min_periods=1).mean()).astype("float32")
    out["ROLLING_STD_3H"] = by_series.transform(
        lambda s: s.rolling(3, min_periods=1).std()).fillna(0).astype("float32")
    out["ROLLING_MEAN_24H"] = by_series.transform(
        lambda s: s.rolling(24, min_periods=1).mean()).astype("float32")

    if dropna:
        out = out.dropna(subset=["LAG_168"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_matrix(
    d: pd.DataFrame,
    feature_names: list[str] = FC_FEATURES,
) -> pd.DataFrame:
    """Select `feature_names` IN ORDER and cast to float32.

    Order matters: the model is fit on this sequence and predicts on it. Passing
    the same columns in a different order would silently feed HOUR_SIN into the
    slot the model learned as PROF_MEAN.

    float32 halves memory on 8.9M rows and is what the notebook trained on.

    Raises on any missing column -- a silently-absent feature is the difference
    between a trustworthy score and not knowing why you got one.
    """
    missing = [c for c in feature_names if c not in d.columns]
    if missing:
        raise KeyError(
            f"missing features {missing}. Did you run add_clock_features, "
            f"calendar_lux.add_holiday_features and attach_profiles?")
    return d[feature_names].astype("float32")
