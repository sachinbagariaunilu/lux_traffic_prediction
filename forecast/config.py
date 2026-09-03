"""Paths, year policy, and the leakage guard.

TRAIN_YEARS is the single source of truth for which years may enter training.
assert_no_leakage() fails the run if data/raw holds anything else, so the
"only 2024 until the model is trained" rule is enforced rather than remembered.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
EXTERNAL = ROOT / "data" / "external"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "metrics"

# The only years allowed into training. Widen deliberately, never by accident.
#
# REVERTED to [2024] on 2026-09-03 after measuring. 2023 is present in data/raw
# and validated (scripts/check_new_year.py passes), but training on it makes the
# model WORSE and four separate rescue attempts failed:
#
#   2024 only                                 MAE 13.95   <- best
#   2023+2024, naive pooling                      14.72
#   2023+2024, matched capacity (early stop)      14.79
#   2023+2024, 2024 rows weighted x4              14.25
#   2023+2024, drop 272 level-shifted series      14.24
#
# Every variant improves monotonically as 2023's influence FALLS, and none
# reaches 13.95 -- so 2023 simply does not describe 2024 traffic well enough to
# be worth its weight. See docs/FINDINGS.md section 4.
#
# The 2023 FILE is still valuable and should stay: it supplies the growth
# measurement (scripts/measure_growth.py) and the full-year evaluation
# (scripts/evaluate_full_year.py --train 2023 --score 2024), which is what
# revealed that January is the second-worst month of the year.
#
# Widen again only with a measurement, never on the assumption that more data
# helps. 2025 stays OUT regardless: it is the only clean holdout.
TRAIN_YEARS = [2024]

# Years PERMITTED to sit in data/raw. Deliberately separate from TRAIN_YEARS,
# because those are two different questions:
#
#   AVAILABLE_YEARS  "is it safe for this file to be here?"   -> holdout policy
#   TRAIN_YEARS      "do we fit on it?"                       -> a model choice
#
# Conflating them meant that deciding NOT to train on 2023 made the leakage
# guard fail on a file that is perfectly safe to keep -- and the only ways out
# were to delete useful data or to disable the guard. Both are worse than
# naming the distinction.
#
# 2025 is absent from this list ON PURPOSE. It is the holdout, it must never sit
# in data/raw, and evaluate_out_of_year() reads it by explicit path instead.
AVAILABLE_YEARS = [2023, 2024]

GROUP = ["POSTE_ID", "DIRECTION", "VEHICULE"]
TARGET = "TRAFFIC_VOLUME"

# Raw wide format carries one column per hour: P00_01 .. P23_24
HOUR_COLS = [f"P{h:02d}_{h + 1:02d}" for h in range(24)]

ID_VARS = ["POSTE_ID", "DATECOM", "DIRECTION", "VEHICULE", "LOCALITE",
           "ROUTE", "SENS", "D1", "D2", "COORD_X", "COORD_Y"]


def raw_file(year: int) -> Path:
    """The single raw CSV for a year. Ambiguity is an error, not a guess."""
    hits = sorted(RAW.glob(f"*{year}*.csv"))
    if not hits:
        raise FileNotFoundError(f"no raw CSV for {year} in {RAW}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous raw CSV for {year}: {[p.name for p in hits]}")
    return hits[0]


def assert_no_leakage(years: list[int] | None = None) -> None:
    """Fail if data/raw contains a year outside the permitted set.

    Checks AVAILABLE_YEARS, not TRAIN_YEARS: the guard exists to keep the
    HOLDOUT off disk, not to police which years a given experiment fits on.
    Pass `years` explicitly to tighten it for a particular run.
    """
    allowed = set(years or AVAILABLE_YEARS)
    stray = {}
    for p in sorted(RAW.glob("*.csv")):
        found = {int(y) for y in re.findall(r"20\d{2}", p.name)}
        if not found:
            raise ValueError(f"cannot infer a year from {p.name}")
        if not found <= allowed:
            stray[p.name] = sorted(found)
    if stray:
        raise AssertionError(
            f"data/raw holds years outside AVAILABLE_YEARS={sorted(allowed)}: "
            f"{stray}. If this is the holdout, read it by explicit path instead "
            f"(see evaluate.evaluate_out_of_year). If it is genuinely safe to "
            f"keep, add it to AVAILABLE_YEARS -- that is separate from deciding "
            f"to train on it.")
    print(f"leakage guard OK - data/raw holds only {sorted(allowed)}")
