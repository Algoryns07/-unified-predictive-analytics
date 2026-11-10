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

## Dataset Usage and Experimental Context

The impact-aware drift detection experiment uses the breast cancer
binary classification dataset provided by scikit-learn.

### Dataset usage

- The dataset is loaded using scikit-learn's built-in dataset loader.
- Features are used for classification and distribution-shift analysis.
- The target contains the binary classification labels.
- A stratified split is used to create training, reference,
  monitoring and future evaluation partitions.

### Experimental scenarios

The experiments evaluate three conditions:

1. No intentional feature shift.
2. Controlled shift in one feature.
3. Controlled shifts in multiple features.

### Data handling

The monitoring and future evaluation partitions are kept separate.
Monitoring data may be used for retraining, while future evaluation
data is reserved for assessing the resulting model.

### Research limitations

These controlled experiments use a benchmark dataset and simulated
feature changes. They should not be interpreted as evidence of
performance on industrial datasets or live production systems.

