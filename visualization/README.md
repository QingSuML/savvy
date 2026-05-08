# Visualization Utilities

This folder contains the cleaned, source-code versions of the visualization
notebooks and ScanNet event-analysis scripts.

## Files

- `support_matrix_behavior_matrix.py`
  - Prediction-reference support matrix computation and plotting.
  - OGA behavior matrix plotting from method CSVs.

- `identity_discovery_reassociation_events.py`
  - ScanNet discovery/reassociation event analysis.
  - Single-method and multi-method discovery/reassociation curves from
    `identity_events.json`.

- `sequence_strips.py`
  - Qualitative RGB, GT, and prediction strips for paper/demo figures.

Reusable visualization logic is provided as source modules; the notebook is a
lightweight usage example.

See `../notebooks/visualization_examples.ipynb` for a lightweight example
notebook with placeholder paths.
