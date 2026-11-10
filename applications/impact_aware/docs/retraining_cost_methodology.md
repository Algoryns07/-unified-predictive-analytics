# Retraining Resource Measurement

## Objective
Measures resources consumed by model training operations.

## Measurements
- Wall-clock execution time
- Process CPU time
- Peak traced Python memory using tracemalloc

## Limitations
- Does not measure total system RAM.
- Native library allocations may not be fully represented.
- Monetary and energy costs are not currently measured.

## Future Extensions
- Measure total process memory.
- Estimate energy and monetary costs.
- Compare retraining resources across datasets and models.
- Forecast future retraining requirements.

## Repeated Resource Measurement and Reproducibility

The resource measurement utility supports repeated execution of a
computational task to reduce dependence on a single timing observation.

### Measurement procedure

1. Execute the same function multiple times using `measure_repeated`.
2. Collect wall-clock time, process CPU time and peak traced memory.
3. Calculate the mean, median and population standard deviation.
4. Export the summary using `export_measurement_summary`.
5. Preserve the exported CSV alongside the experiment results.

### Recommended experimental practice

- Use at least five repetitions for exploratory measurements.
- Keep the input data and function configuration consistent.
- Record the Python environment and relevant package versions.
- Avoid comparing measurements from substantially different hardware
  or system-load conditions.
- Report both central tendency and variability.
- Treat traced Python memory as an estimate of Python-managed allocations,
  not total operating-system memory consumption.

### Interpretation

Mean and median summarize typical resource consumption, while standard
deviation describes variation between repeated executions.

These measurements provide an operational resource-cost perspective.
They do not independently establish energy consumption or monetary cost.
