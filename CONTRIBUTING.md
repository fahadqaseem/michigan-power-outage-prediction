# Contributing

Thanks for helping improve this Michigan power-outage research project.

## Good first contributions

- Replace the prototype's overlapping random-window split with walk-forward validation.
- Add precision-recall curves, calibrated probabilities, and threshold selection.
- Compare logistic regression, gradient boosting, and graph-based models on identical splits.
- Add weather forecasts without using outage-derived features at prediction time.
- Test county adjacency and station-to-county matching.

## Local workflow

1. Create a virtual environment and install `requirements-dev.txt`.
2. Run `pytest` before opening a pull request.
3. Keep raw downloads, credentials, model weights, and notebook checkpoints out of Git.
4. Explain the data period, split strategy, and imbalance-aware metrics in model changes.

Please keep pull requests focused and include a short description of how the change was verified.
