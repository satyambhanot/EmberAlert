"""Data loading, target definition and feature engineering for the wildfire project.

The model supports one decision: right after Alberta Wildfire first assesses a
new fire, should the fire be treated as one that will become large? To match
that decision point, the model may only use information known at the initial
assessment. Every column recorded later is listed in ``LEAKAGE_COLUMNS`` and is
never used as a feature.

Typical use (from the repository root)::

    from src.features import load_raw_data, make_target, build_features, build_preprocessor
    raw = load_raw_data()
    X, y = build_features(raw), make_target(raw)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "fp-historical-wildfire-data-2006-2025.csv"

# Size classes that count as a "large" fire (final area burned > 40 ha).
LARGE_SIZE_CLASSES = ["D", "E"]

# Columns that must never be used as features, with the reason for each.
# Timing comes from the official data dictionary and was checked against the data.
LEAKAGE_COLUMNS = {
    # The target itself
    "CURRENT_SIZE": "final area burned (the target is derived from it)",
    "SIZE_CLASS": "final size class (the target)",
    # Assigned or investigated after the fact
    "FIRE_NAME": "only significant fires are named (43% of named fires are large vs 2% of unnamed)",
    "TRUE_CAUSE": "set by later investigation; missing far more often for large fires",
    "ACTIVITY_CLASS": "set by later investigation",
    "RESPONSIBLE_GROUP": "set by later investigation",
    "INDUSTRY_IDENTIFIER": "set by later investigation",
    # The response itself (happens after the decision we are modelling)
    "DISPATCHED_RESOURCE": "the dispatcher's own judgement of the fire, i.e. the decision being modelled",
    "INITIAL_ACTION_BY": "the suppression response (air tankers are sent to fires already judged dangerous)",
    "IA_ARRIVAL_AT_FIRE_DATE": "suppression response",
    "IA_ACCESS": "suppression response",
    "FIRE_FIGHTING_START_DATE": "suppression response",
    "FIRE_FIGHTING_START_SIZE": "size after the assessment",
    "BUCKETING_ON_FIRE": "suppression response",
    "DISTANCE_FROM_WATER_SOURCE": "suppression response",
    "FIRST_BUCKET_DROP_DATE": "suppression response",
    # Fire status updates recorded after the assessment
    "FIRST_BH_DATE": "status update after assessment",
    "FIRST_BH_SIZE": "status update after assessment",
    "FIRST_UC_DATE": "status update after assessment",
    "FIRST_UC_SIZE": "status update after assessment",
    "FIRST_TO_DATE": "status update after assessment",
    "FIRST_TO_SIZE": "status update after assessment",
    "FIRST_EX_DATE": "status update after assessment",
    "FIRST_EX_SIZE_PERIMETER": "status update after assessment",
}

# Columns that are not leakage but are still left out, with the reason for each.
UNUSED_COLUMNS = {
    "YEAR": "kept out of the features; useful for a time-based train/test check",
    "FIRE_NUMBER": "identifier; only its first letter (forest area) is used",
    "GENERAL_CAUSE": "final cause may be revised after investigation; only Lightning vs Human/other is used",
    "FIRE_START_DATE": "often an estimate revised later",
    "DISCOVERED_DATE": "21% missing; REPORTED_DATE is used instead",
    "DISCOVERED_SIZE": "99% missing",
    "DETECTION_AGENT": "fine-grained version of DETECTION_AGENT_TYPE (mostly identifiers)",
    "DISPATCH_DATE": "covered by REPORTED_DATE and ASSESSMENT_DATETIME",
    "START_FOR_FIRE_DATE": "covered by REPORTED_DATE and ASSESSMENT_DATETIME",
    "ASSESSMENT_RESOURCE": "name/position of the assessor",
}

# Features available at the initial assessment (after build_features()).
CATEGORICAL_FEATURES = [
    "forest_area",        # first letter of FIRE_NUMBER (C = Calgary, E = Edson, ...)
    "cause_group",        # Lightning vs Human/other (coarse cause is known at assessment)
    "fire_origin",        # land ownership
    "detection_agent",    # LKT lookout, AIR air patrol, GRP ground, UNP unplanned, CAM camera
    "fuel_type",          # vegetation the fire is burning in
    "weather",            # weather conditions over the fire
    "fire_type",          # ground / surface / crown
    "slope_position",     # position on slope
    "wind_direction",
    "report_month",
]
NUMERIC_FEATURES = [
    "spread_rate_m_per_min",
    "temperature_c",
    "relative_humidity",
    "wind_speed_kmh",
    "latitude",
    "longitude",
    "report_hour",
    "hours_report_to_assessment",
    "log_assessment_ha",  # log(1 + size at assessment); raw size is extremely skewed
]


def load_raw_data(path=DATA_PATH):
    """Read the raw wildfire CSV into a DataFrame.

    Args:
        path: location of the CSV. Defaults to the copy in ``data/``.

    Returns:
        pandas.DataFrame with all 50 original columns.
    """
    return pd.read_csv(path, low_memory=False)


def make_target(raw):
    """Return the binary target: 1 if the fire ended as size class D or E (> 40 ha).

    Args:
        raw: DataFrame from :func:`load_raw_data`.

    Returns:
        pandas.Series of 0/1 integers named ``large_fire``.
    """
    return raw["SIZE_CLASS"].isin(LARGE_SIZE_CLASSES).astype(int).rename("large_fire")


def _clean_text(series):
    """Strip whitespace and turn empty strings into NaN so the imputer sees them as missing."""
    cleaned = series.where(series.isna(), series.astype(str).str.strip())
    return cleaned.replace("", np.nan).astype(object)


def build_features(raw):
    """Build the model's feature matrix using only information known at the initial assessment.

    Leakage columns are never read. Missing values are left as NaN on purpose,
    so that imputation happens inside the scikit-learn pipeline (fitted on
    training folds only).

    Args:
        raw: DataFrame from :func:`load_raw_data`.

    Returns:
        pandas.DataFrame with CATEGORICAL_FEATURES + NUMERIC_FEATURES columns.
    """
    reported = pd.to_datetime(raw["REPORTED_DATE"], errors="coerce")
    assessed = pd.to_datetime(raw["ASSESSMENT_DATETIME"], errors="coerce")

    X = pd.DataFrame(index=raw.index)
    X["forest_area"] = raw["FIRE_NUMBER"].str[0]
    X["cause_group"] = np.where(raw["GENERAL_CAUSE"] == "Lightning", "Lightning", "Human/other")
    X["fire_origin"] = raw["FIRE_ORIGIN"]
    X["detection_agent"] = raw["DETECTION_AGENT_TYPE"]
    X["fuel_type"] = raw["FUEL_TYPE"]
    X["weather"] = raw["WEATHER_CONDITIONS_OVER_FIRE"]
    X["fire_type"] = raw["FIRE_TYPE"]
    X["slope_position"] = raw["FIRE_POSITION_ON_SLOPE"]
    X["wind_direction"] = raw["WIND_DIRECTION"]
    X["report_month"] = reported.dt.month.map(lambda m: f"{m:02.0f}", na_action="ignore")
    for col in CATEGORICAL_FEATURES:
        X[col] = _clean_text(X[col])

    X["spread_rate_m_per_min"] = raw["FIRE_SPREAD_RATE"]
    X["temperature_c"] = raw["TEMPERATURE"]
    X["relative_humidity"] = raw["RELATIVE_HUMIDITY"]
    X["wind_speed_kmh"] = raw["WIND_SPEED"]
    X["latitude"] = raw["LATITUDE"]
    X["longitude"] = raw["LONGITUDE"]
    X["report_hour"] = reported.dt.hour
    X["hours_report_to_assessment"] = (assessed - reported).dt.total_seconds() / 3600
    X["log_assessment_ha"] = np.log1p(raw["ASSESSMENT_HECTARES"])
    return X[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def build_preprocessor(scale_numeric=True):
    """Create the ColumnTransformer shared by every model.

    Numeric columns: median imputation, then standard scaling (needed by
    LogisticRegression, LinearSVC, SVC and KNN; skipped for tree models).
    Categorical columns: missing values become their own "missing" category,
    then one-hot encoding. Categories seen fewer than 20 times are grouped
    together, and categories never seen in training are ignored.

    Args:
        scale_numeric: set False for tree-based models, which do not need scaling.

    Returns:
        sklearn.compose.ColumnTransformer (unfitted).
    """
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale_numeric:
        numeric_steps.append(StandardScaler())
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
    )
    return ColumnTransformer([
        ("num", make_pipeline(*numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
