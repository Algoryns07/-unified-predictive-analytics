# Dataset policy

The starter project uses datasets bundled with scikit-learn, so no external download is required.

For dissertation experiments, add a public dataset only after checking its licence and suitability. Document:
- official dataset source and version/access date
- licence and permitted use
- target and feature columns
- missing-value and encoding decisions
- split method (prefer chronological splits for time-dependent data)
- monitoring-window construction and simulated shift assumptions

Do not commit private, restricted, or large raw data files.
