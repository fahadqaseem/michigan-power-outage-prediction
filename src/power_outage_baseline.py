"""Leakage-aware baseline for Michigan extreme power-outage risk.

The original project explored a spatio-temporal graph neural network. This
small baseline provides a reproducible reference point using only information
available before the outage label is calculated. It intentionally excludes
``customers_out`` and ``total_customers`` from the predictors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "outage_risk"
NON_FEATURE_COLUMNS = {"date", TARGET, "customers_out", "total_customers"}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the processed county-day dataset."""
    frame = pd.read_csv(path, low_memory=False)
    required = {"county", "date", TARGET, "customers_out"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1")
    return frame.sort_values("date").reset_index(drop=True)


def choose_features(frame: pd.DataFrame, missing_limit: float = 0.95) -> list[str]:
    """Select usable predictors without using label-defining outage fields."""
    candidates = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]
    return [
        column
        for column in candidates
        if column == "county"
        or (
            pd.api.types.is_numeric_dtype(frame[column])
            and frame[column].isna().mean() <= missing_limit
            and frame[column].nunique(dropna=True) > 1
        )
    ]


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on unique dates so future records never enter the training set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    dates = np.sort(frame["date"].unique())
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - test_fraction))))
    cutoff = pd.Timestamp(dates[split_index])
    train = frame[frame["date"] < cutoff].copy()
    test = frame[frame["date"] >= cutoff].copy()

    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError(
            "Both chronological partitions need positive and negative examples; "
            "choose another test fraction."
        )
    return train, test, cutoff


def build_pipeline(features: list[str]) -> Pipeline:
    """Build an imputation, scaling, encoding, and balanced logistic model."""
    categorical = [column for column in features if column == "county"]
    numeric = [column for column in features if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    """Return imbalance-aware metrics and a thresholded diagnostic report."""
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }


def run(data_path: Path, test_fraction: float = 0.20) -> dict[str, Any]:
    """Train the baseline and return dataset, split, and model diagnostics."""
    frame = load_dataset(data_path)
    features = choose_features(frame)
    train, test, cutoff = chronological_split(frame, test_fraction)

    model = build_pipeline(features)
    model.fit(train[features], train[TARGET])
    probabilities = model.predict_proba(test[features])[:, 1]

    return {
        "dataset": {
            "rows": len(frame),
            "counties": int(frame["county"].nunique()),
            "start_date": frame["date"].min().date().isoformat(),
            "end_date": frame["date"].max().date().isoformat(),
            "positive_examples": int(frame[TARGET].sum()),
            "positive_rate": float(frame[TARGET].mean()),
        },
        "split": {
            "cutoff": cutoff.date().isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_positives": int(train[TARGET].sum()),
            "test_positives": int(test[TARGET].sum()),
        },
        "features": features,
        "metrics": evaluate(test[TARGET], probabilities),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/merged_weather_outage_data.csv"),
        help="Path to the processed county-day CSV.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of the latest unique dates reserved for testing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.data, args.test_fraction)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

