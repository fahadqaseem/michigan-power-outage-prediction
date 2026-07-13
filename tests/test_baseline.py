from pathlib import Path

import pandas as pd
import pytest

from src.power_outage_baseline import chronological_split, choose_features, load_dataset


DATA_PATH = Path("data/processed/merged_weather_outage_data.csv")


def test_published_dataset_contract() -> None:
    frame = load_dataset(DATA_PATH)

    assert len(frame) == 150_963
    assert frame["county"].nunique() == 83
    assert frame["outage_risk"].sum() == 49
    assert frame["date"].min() == pd.Timestamp("2020-01-01")
    assert frame["date"].max() == pd.Timestamp("2024-12-31")


def test_label_defining_columns_are_never_features() -> None:
    frame = load_dataset(DATA_PATH)
    features = choose_features(frame)

    assert "county" in features
    assert "customers_out" not in features
    assert "total_customers" not in features
    assert "outage_risk" not in features


def test_split_is_strictly_chronological() -> None:
    frame = load_dataset(DATA_PATH)
    train, test, cutoff = chronological_split(frame)

    assert train["date"].max() < cutoff
    assert test["date"].min() >= cutoff
    assert train["outage_risk"].nunique() == 2
    assert test["outage_risk"].nunique() == 2


def test_split_rejects_invalid_fraction() -> None:
    frame = load_dataset(DATA_PATH)

    with pytest.raises(ValueError, match="between 0 and 1"):
        chronological_split(frame, test_fraction=1.0)

