# MetroPT Quality Checks

`quality_checks.py` contains lightweight, reusable data-quality functions.

Functions:

- `find_signal_overlap_transitions`: identifies simultaneous COMP/DV_eletric
  states with immediate one-second neighbours.
- `calculate_gps_agreement`: calculates reported-versus-coordinate-derived
  speed agreement using only valid comparisons.
- `group_high_speed_episodes`: groups consecutive observations using a
  configurable timestamp gap.
- `identify_timestamp_gaps`: identifies irregular sampling intervals.

The functions operate on supplied DataFrames or timestamp sequences.
They do not load the raw MetroPT dataset.

Tests use small synthetic datasets and can be executed with:

`python -m pytest metropt-industrial-ai/tests -v`

These functions support exploratory quality assessment. They do not
establish sensor ground truth or prove physical equipment faults.
