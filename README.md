# Michigan Power Outage Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN%20prototype-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An exploratory machine-learning project that combines county-level power-outage records, daily weather observations, and Michigan county geography to study extreme outage risk. The work progresses from data collection and event preprocessing to a spatio-temporal graph neural network (STGNN) prototype, with a reproducible leakage-aware logistic-regression baseline for comparison.

> **Research status:** this is a portfolio and learning project, not an operational outage-warning system. Extreme outages are exceptionally rare in the published dataset, so accuracy alone is misleading.

## Why this project matters

Severe weather can disrupt electric service across neighboring counties and over multiple days. That makes the problem both spatial and temporal. This project explores whether weather signals and county relationships can help identify county-days at risk of an extreme outage, defined here as at least 50,000 customers without power.

## What I built

- Cleaned and aggregated high-frequency county outage observations into outage events.
- Joined outage records with NOAA daily weather measurements for all 83 Michigan counties.
- Used U.S. Census county boundaries to explore geographic adjacency.
- Engineered weather, calendar, outage-rate, duration, and cyclical time features.
- Prototyped a multi-task STGNN using graph convolutions, recurrent layers, and attention.
- Added a reproducible logistic-regression baseline with chronological validation and no outage-derived predictors.

## Dataset snapshot

| Attribute | Value |
|---|---:|
| Time range | 2020-01-01 to 2024-12-31 |
| County-day rows | 150,963 |
| Michigan counties | 83 |
| Weather/outage columns | 37 |
| Extreme county-days | 49 (0.0325%) |
| Extreme-outage threshold | 50,000 customers out |

The class imbalance is central to the problem: a model that always predicts “no extreme outage” would exceed 99.9% accuracy. For that reason, the baseline reports average precision, ROC AUC, recall, precision, F1, and a confusion matrix.

## Reproducible baseline result

The chronological holdout reserves 2024 for testing and trains on 2020–2022 observations (the processed dataset contains no 2023 records). At the default 0.5 threshold:

| Metric | Result |
|---|---:|
| Average precision | 0.0887 |
| ROC AUC | 0.7968 |
| Extreme-event recall | 0.8000 (4/5) |
| Extreme-event precision | 0.0044 |
| False-positive county-days | 910 |

This is a diagnostic reference point, not a deployment claim. With only five positive test examples, the estimates are unstable; the low precision also shows why probability calibration and operational threshold selection are essential.

## Project workflow

```text
EAGLE-I outage records ──┐
                        ├─> county/day aggregation ─> merged dataset ─> models
NOAA GHCN-D weather ─────┤                                  │
                        │                                  ├─> chronological baseline
U.S. Census boundaries ─┘                                  └─> STGNN prototype
```

## Repository structure

```text
.
├── data/processed/merged_weather_outage_data.csv
├── notebooks/
│   ├── 01_outage_event_preprocessing.ipynb
│   └── 02_spatiotemporal_model_prototype.ipynb
├── src/power_outage_baseline.py
├── tests/test_baseline.py
├── .github/workflows/ci.yml
├── CONTRIBUTING.md
├── CITATION.cff
├── requirements.txt
└── requirements-dev.txt
```

The notebooks preserve the original research path. The command-line baseline is the recommended reproducible entry point. Raw downloads, exploratory duplicates, trained weights, and local credentials are intentionally excluded from Git.

## Quick start

```bash
git clone https://github.com/fahadqaseem/michigan-power-outage-prediction.git
cd michigan-power-outage-prediction
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/power_outage_baseline.py
```

To save the evaluation report:

```bash
python3 src/power_outage_baseline.py --output artifacts/baseline_metrics.json
```

For notebook exploration, install the optional stack:

```bash
python3 -m pip install -r requirements-notebooks.txt
jupyter lab
```

To run the quality checks:

```bash
python3 -m pip install -r requirements-dev.txt
pytest
ruff check src tests
```

## Modeling notes and honest limitations

The original prototype reported approximately 99.9% extreme-event classification accuracy. That result should not be treated as a validated performance claim because the target is extremely imbalanced and the prototype randomly split overlapping temporal windows. Nearby windows can therefore appear in both training and validation sets.

The published baseline addresses the most important evaluation risks:

- It splits records chronologically, keeping future dates out of training.
- It excludes `customers_out` and `total_customers`, which directly define or reveal the label.
- It balances class weights during training.
- It emphasizes precision-recall metrics rather than raw accuracy.

Further work should use walk-forward validation, probability calibration, event-level scoring, uncertainty estimates, and comparison against persistence and weather-only baselines. The processed data also contains sparse weather-event fields and only 49 positive county-days, concentrated in a few populous counties.

## Data sources

- [EAGLE-I Power Outage Data 2014–2022](https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1), Oak Ridge National Laboratory. County-level observations are available at 15-minute intervals.
- [Global Historical Climatology Network Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), NOAA National Centers for Environmental Information.
- [2024 Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/2024/geo/carto-boundary-file.html), U.S. Census Bureau.

Users should review each upstream source's terms and documentation before redistributing derived data. The processed CSV is included for reproducibility and should be cited together with the original sources.

## Contributing

Ideas, issue reports, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for high-impact next steps, including walk-forward validation, calibrated probabilities, and stronger graph baselines.

## Keywords

`power outage prediction` · `machine learning` · `graph neural network` · `spatio-temporal forecasting` · `extreme weather` · `energy resilience` · `critical infrastructure` · `PyTorch Geometric` · `NOAA weather data` · `EAGLE-I` · `Michigan`

## License

Code is available under the [MIT License](LICENSE). External datasets remain subject to their original providers' terms.
