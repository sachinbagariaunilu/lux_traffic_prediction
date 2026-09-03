"""Luxembourg holiday calendar and the three holiday features."""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

_PUBLIC_JSON = config.EXTERNAL / "public_holidays.json"
_SCHOOL_JSON = config.EXTERNAL / "school_holidays.json"

HOLIDAY_FEATURES = ["IS_PUBLIC_HOLIDAY", "IS_SCHOOL_HOLIDAY", "DAYS_TO_HOLIDAY"]


@lru_cache(maxsize=1)
def _public_spec() -> dict:
    return json.loads(_PUBLIC_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _school_spec() -> dict:
    return json.loads(_SCHOOL_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Easter and public holidays
# --------------------------------------------------------------------------


def easter_sunday(year: int) -> dt.date:
    """Easter Sunday via the anonymous Gregorian computus.

    Fixtures: 2024-03-31, 2025-04-20, 2026-04-05, 2027-03-28, 2028-04-16.
    """
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


@lru_cache(maxsize=64)
def _holidays_for_year(year: int) -> tuple[dt.date, ...]:
    """One year's statutory holidays, deduplicated and sorted."""
    spec = _public_spec()["rules"]
    easter = easter_sunday(year)
    dates = {dt.date(year, r["month"], r["day"]) for r in spec["fixed"]}
    # set union does the dedupe: 2024 collapses Ascension onto Europe Day
    dates |= {easter + dt.timedelta(r["offset_days"]) for r in spec["easter_offsets"]}
    return tuple(sorted(dates))


def public_holidays(years: list[int]) -> pd.DatetimeIndex:
    """The Luxembourg statutory holidays for `years`, DEDUPLICATED and sorted.

    Returns 10 dates for 2024 (Ascension == Europe Day) and 11 for 2025-2029.
    A result of 11 for 2024 means the dedupe is broken.
    """
    out: set[dt.date] = set()
    for y in years:
        out.update(_holidays_for_year(int(y)))
    return pd.DatetimeIndex(sorted(out))


def recorded_holidays(years: list[int]) -> pd.DatetimeIndex:
    """The dates RECORDED in public_holidays.json, for `years`.

    The fixture side of the generator check. Must never call
    public_holidays(), or comparing the two becomes a tautology that passes
    even when the generator is wrong.

    2026-2028 were verified against the ITM published table.
    """
    spec = _public_spec()["years"]
    dates: list[str] = []
    for y in years:
        key = str(int(y))
        if key not in spec:
            raise KeyError(f"no recorded fixtures for {key}")
        dates += spec[key]["dates"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(dates)))


# --------------------------------------------------------------------------
# School holidays
# --------------------------------------------------------------------------


def school_ranges(
    years: list[int] | None = None,
    *,
    include_single_days: bool = False,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Flat list of inclusive (start, end) school-holiday ranges, sorted.

    Flattens the school-year nesting in school_holidays.json. Pass `years` to
    keep only ranges INTERSECTING those calendar years (a Christmas range spans
    two years, so it belongs to both), or None for all.

    Always the OFFICIAL dates from the grand-ducal regulation. There is no
    legacy mode: the notebook's approximate ranges had 29 mislabelled days and
    are kept in the JSON purely as a record of what was wrong, never as a code
    path something could accidentally select.

    include_single_days=True adds saint_nicolas (6 Dec 2027, 2028) as one-day
        ranges. Off by default: it closes ELEMENTARY schools only, so the
        traffic effect is partial.
    """
    spec = _school_spec()

    pairs = []
    for block in spec["school_years"].values():
        pairs += [(pd.Timestamp(a), pd.Timestamp(b))
                  for a, b in block["holidays"].values()]
        if include_single_days:
            pairs += [(pd.Timestamp(d), pd.Timestamp(d))
                      for d in block.get("single_days_off", {}).values()]

    if years is not None:
        wanted = {int(y) for y in years}
        # intersects, not "starts in" -- Christmas spans a year boundary
        pairs = [(a, b) for a, b in pairs
                 if wanted & set(range(a.year, b.year + 1))]

    return sorted(pairs)


def calendar_coverage() -> tuple[pd.Timestamp, pd.Timestamp]:
    """(first, last) date whose holiday status is actually known.

    Lower bound is the earliest school-year start; upper bound the latest
    holiday end. Public holidays impose no limit -- they are computable for any
    year -- so the school data is always what binds.
    """
    spec = _school_spec()["school_years"]
    first = min(pd.Timestamp(b["school_year"][0]) for b in spec.values())
    last = max(pd.Timestamp(e)
               for b in spec.values() for _, e in b["holidays"].values())
    return first, last


def assert_covers(start, end=None) -> None:
    """Raise if [start, end] extends beyond calendar_coverage().

    The guard that turns Section 3's silent failure into an error. Without it
    forecast_dates() happily returns holiday-blind numbers while printing a
    confident expected-error figure.
    """
    a = pd.Timestamp(start).normalize()
    b = pd.Timestamp(end).normalize() if end is not None else a
    first, last = calendar_coverage()
    if a < first or b > last:
        raise ValueError(
            f"calendar covers {first:%Y-%m-%d} .. {last:%Y-%m-%d}; refusing to "
            f"predict {a:%Y-%m-%d} .. {b:%Y-%m-%d}. All three holiday features "
            f"would be constant there, silently degrading the model to a "
            f"calendar-blind one. Extend data/external/school_holidays.json "
            f"from the MENJE regulation (men.public.lu/fr/vacances-scolaires)."
        )


# --------------------------------------------------------------------------
# The three model features
# --------------------------------------------------------------------------


def add_holiday_features(
    d: pd.DataFrame,
    *,
    ts_col: str = "TIME_STAMP",
    signed_days: bool = False,
    cap: int = 7,
    holiday_years: list[int] | None = None,
) -> pd.DataFrame:
    """Attach IS_PUBLIC_HOLIDAY, IS_SCHOOL_HOLIDAY, DAYS_TO_HOLIDAY.

    Idempotent: drops any pre-existing copies first, so re-running cannot
    produce the _x/_y merge debris the notebook suffered from.

    DAYS_TO_HOLIDAY is computed on the UNIQUE dates and mapped back, never
    broadcast over all rows. Measured on 8.7M rows: 3,590 numbers instead of
    87,218,880 -- 13x faster, 10x less peak memory, identical results.

    signed_days=False (default) uses abs(distance) capped at `cap`, so "3 days
        before Christmas" == "3 days after". That conflation is finding #4 and
        is probably a flaw -- but it is UNMEASURED, so it stays the default and
        the change gets tested on its own rather than bundled with others.
    signed_days=True keeps the sign: negative means the nearest holiday is
        ahead, positive means it has passed.
    holiday_years   which years' public holidays to consider. Default is the
        years present in the data plus one either side, which is what you want:
        a 30 December row should see 1 January as its nearest holiday. Narrowing
        it changes DAYS_TO_HOLIDAY near year boundaries -- with 2025 excluded,
        2024-12-30 scores 4 (to 26 Dec) instead of 2 (to 1 Jan). Only narrow it
        deliberately, and say why.
    """
    d = d.drop(columns=[c for c in HOLIDAY_FEATURES if c in d.columns])

    dates = d[ts_col].dt.normalize()

    if holiday_years is None:
        present = dates.dt.year.unique().tolist()
        holiday_years = sorted({int(y) + off for y in present for off in (-1, 0, 1)})
    ph = public_holidays(holiday_years)

    d["IS_PUBLIC_HOLIDAY"] = dates.isin(ph).astype("int8")

    ranges = school_ranges()
    in_school = np.zeros(len(d), dtype=bool)
    for a, b in ranges:
        in_school |= ((dates >= a) & (dates <= b)).to_numpy()
    d["IS_SCHOOL_HOLIDAY"] = in_school.astype("int8")

    # --- the unique-dates trick: ~360 dates, not 8.7M rows
    uniq = pd.DatetimeIndex(dates.unique())
    delta = (uniq.to_numpy()[:, None] - ph.to_numpy()[None, :]) / np.timedelta64(1, "D")
    nearest = np.abs(delta).argmin(axis=1)
    signed = delta[np.arange(len(uniq)), nearest]
    value = np.clip(signed, -cap, cap) if signed_days else np.minimum(np.abs(signed), cap)
    d["DAYS_TO_HOLIDAY"] = dates.map(
        pd.Series(value, index=uniq)).astype("float32")

    return d


def add_term_split_profiles_key(d: pd.DataFrame) -> pd.DataFrame:
    """Add IS_SCHOOL_TERM, the key for term-split profiles.

    A GROUPING KEY, not a model feature -- the model already has
    IS_SCHOOL_HOLIDAY, and this is its exact inverse. Adding both would be
    duplicate information.

    Used in two places: features.build_profiles(split_by_term=True) groups on it
    when building the averages, and prediction uses it to pick which average
    applies. It never becomes a column the model reads.

    Why: PROF_DOW_HOUR currently blends term-time and holiday days for the same
    (counter, weekday, hour), producing a mean that fits neither. Finding #6.

    Measured on 2024, profile used as a DIRECT predictor:
        blended                  14.71
        split by school term     13.23   (10.1% better, in-sample)
    An earlier version of this docstring claimed ~19% by attributing the whole
    oracle A->B gap here; about half of that gap actually comes from splitting
    on PUBLIC holidays, not school term. The gain to the model is smaller again,
    because it already has the flags -- measure it, do not assume it.

    Sparsity risk was checked: splitting doubles the groups (177,744 ->
    355,488) and halves the median group size (51 -> 21 observations), but only
    0.1% of rows land in a group with under 10 observations. Safe.

    Requires IS_SCHOOL_HOLIDAY, so call add_holiday_features() first.
    """
    if "IS_SCHOOL_HOLIDAY" not in d.columns:
        raise ValueError("call add_holiday_features() before this")
    d = d.copy()
    d["IS_SCHOOL_TERM"] = (1 - d["IS_SCHOOL_HOLIDAY"]).astype("int8")
    return d
