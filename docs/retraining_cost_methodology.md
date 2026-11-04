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
