# MetroPT Temporal Robustness Assessment

## Experimental design

- Three expanding chronological training windows.
- Three successive, non-overlapping test periods.
- Three random seeds: 17, 42 and 73.
- Three regression models and a persistence baseline.
- Targets: Oil_temperature, TP3 and Motor_current.
- Metrics: MAE, RMSE and R².
- Lag and rolling features do not cross recording gaps.
- No model selection is performed using test performance.

## Aggregate results

| Target | Model | Experiments | Mean RMSE | RMSE Std | Mean R² | Persistence RMSE | RMSE reduction (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Motor_current | Decision Tree | 9 | 0.651627 | 0.144007 | 0.891735 | 1.499530 | 56.54 |
| Motor_current | HistGradientBoosting | 9 | 0.577307 | 0.071835 | 0.917444 | 1.499530 | 61.50 |
| Motor_current | Linear Regression | 9 | 1.189686 | 0.015761 | 0.651727 | 1.499530 | 20.66 |
| Oil_temperature | Decision Tree | 9 | 0.888563 | 0.158944 | 0.957421 | 2.305208 | 61.45 |
| Oil_temperature | HistGradientBoosting | 9 | 0.817500 | 0.174357 | 0.964004 | 2.305208 | 64.54 |
| Oil_temperature | Linear Regression | 9 | 1.104896 | 0.111844 | 0.933730 | 2.305208 | 52.07 |
| TP3 | Decision Tree | 9 | 0.135388 | 0.015480 | 0.948795 | 0.389377 | 65.23 |
| TP3 | HistGradientBoosting | 9 | 0.141095 | 0.024893 | 0.942951 | 0.389377 | 63.76 |
| TP3 | Linear Regression | 9 | 0.198370 | 0.000223 | 0.890956 | 0.389377 | 49.05 |

## Per-window results

| Target | Window | Model | Seed | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|---:|
| Oil_temperature | Window 1 | Persistence | Baseline | 1.443454 | 2.126644 | 0.696385 |
| Oil_temperature | Window 1 | Linear Regression | 17 | 0.694951 | 0.995547 | 0.933464 |
| Oil_temperature | Window 1 | Decision Tree | 17 | 0.561666 | 0.775488 | 0.959628 |
| Oil_temperature | Window 1 | HistGradientBoosting | 17 | 0.511287 | 0.700658 | 0.967043 |
| Oil_temperature | Window 1 | Linear Regression | 42 | 0.694951 | 0.995547 | 0.933464 |
| Oil_temperature | Window 1 | Decision Tree | 42 | 0.561666 | 0.775488 | 0.959628 |
| Oil_temperature | Window 1 | HistGradientBoosting | 42 | 0.503291 | 0.690888 | 0.967956 |
| Oil_temperature | Window 1 | Linear Regression | 73 | 0.694951 | 0.995547 | 0.933464 |
| Oil_temperature | Window 1 | Decision Tree | 73 | 0.565115 | 0.783672 | 0.958771 |
| Oil_temperature | Window 1 | HistGradientBoosting | 73 | 0.490497 | 0.676909 | 0.969239 |
| Oil_temperature | Window 2 | Persistence | Baseline | 1.558233 | 2.302204 | 0.693724 |
| Oil_temperature | Window 2 | Linear Regression | 17 | 0.705813 | 1.071758 | 0.933623 |
| Oil_temperature | Window 2 | Decision Tree | 17 | 0.464359 | 0.787070 | 0.964203 |
| Oil_temperature | Window 2 | HistGradientBoosting | 17 | 0.399956 | 0.708744 | 0.970973 |
| Oil_temperature | Window 2 | Linear Regression | 42 | 0.705813 | 1.071758 | 0.933623 |
| Oil_temperature | Window 2 | Decision Tree | 42 | 0.464359 | 0.787070 | 0.964203 |
| Oil_temperature | Window 2 | HistGradientBoosting | 42 | 0.402418 | 0.719896 | 0.970052 |
| Oil_temperature | Window 2 | Linear Regression | 73 | 0.705813 | 1.071758 | 0.933623 |
| Oil_temperature | Window 2 | Decision Tree | 73 | 0.464359 | 0.787070 | 0.964203 |
| Oil_temperature | Window 2 | HistGradientBoosting | 73 | 0.398234 | 0.712251 | 0.970685 |
| Oil_temperature | Window 3 | Persistence | Baseline | 1.629629 | 2.486777 | 0.738100 |
| Oil_temperature | Window 3 | Linear Regression | 17 | 0.749293 | 1.247382 | 0.934104 |
| Oil_temperature | Window 3 | Decision Tree | 17 | 0.617877 | 1.100403 | 0.948718 |
| Oil_temperature | Window 3 | HistGradientBoosting | 17 | 0.549946 | 1.049698 | 0.953335 |
| Oil_temperature | Window 3 | Linear Regression | 42 | 0.749293 | 1.247382 | 0.934104 |
| Oil_temperature | Window 3 | Decision Tree | 42 | 0.617877 | 1.100403 | 0.948718 |
| Oil_temperature | Window 3 | HistGradientBoosting | 42 | 0.548626 | 1.051173 | 0.953204 |
| Oil_temperature | Window 3 | Linear Regression | 73 | 0.749293 | 1.247382 | 0.934104 |
| Oil_temperature | Window 3 | Decision Tree | 73 | 0.617877 | 1.100403 | 0.948718 |
| Oil_temperature | Window 3 | HistGradientBoosting | 73 | 0.549244 | 1.047288 | 0.953549 |
| TP3 | Window 1 | Persistence | Baseline | 0.262438 | 0.384332 | 0.567139 |
| TP3 | Window 1 | Linear Regression | 17 | 0.128387 | 0.198379 | 0.884674 |
| TP3 | Window 1 | Decision Tree | 17 | 0.073332 | 0.130725 | 0.949921 |
| TP3 | Window 1 | HistGradientBoosting | 17 | 0.103762 | 0.191604 | 0.892418 |
| TP3 | Window 1 | Linear Regression | 42 | 0.128387 | 0.198379 | 0.884674 |
| TP3 | Window 1 | Decision Tree | 42 | 0.073332 | 0.130725 | 0.949921 |
| TP3 | Window 1 | HistGradientBoosting | 42 | 0.067239 | 0.119042 | 0.958473 |
| TP3 | Window 1 | Linear Regression | 73 | 0.128387 | 0.198379 | 0.884674 |
| TP3 | Window 1 | Decision Tree | 73 | 0.073332 | 0.130725 | 0.949921 |
| TP3 | Window 1 | HistGradientBoosting | 73 | 0.087788 | 0.160527 | 0.924486 |
| TP3 | Window 2 | Persistence | Baseline | 0.278555 | 0.402409 | 0.560998 |
| TP3 | Window 2 | Linear Regression | 17 | 0.130053 | 0.198622 | 0.893049 |
| TP3 | Window 2 | Decision Tree | 17 | 0.049384 | 0.120307 | 0.960761 |
| TP3 | Window 2 | HistGradientBoosting | 17 | 0.047046 | 0.118256 | 0.962088 |
| TP3 | Window 2 | Linear Regression | 42 | 0.130053 | 0.198622 | 0.893049 |
| TP3 | Window 2 | Decision Tree | 42 | 0.049384 | 0.120307 | 0.960761 |
| TP3 | Window 2 | HistGradientBoosting | 42 | 0.046725 | 0.119773 | 0.961109 |
| TP3 | Window 2 | Linear Regression | 73 | 0.130053 | 0.198622 | 0.893049 |
| TP3 | Window 2 | Decision Tree | 73 | 0.049384 | 0.120307 | 0.960761 |
| TP3 | Window 2 | HistGradientBoosting | 73 | 0.046724 | 0.119643 | 0.961193 |
| TP3 | Window 3 | Persistence | Baseline | 0.252093 | 0.381389 | 0.611385 |
| TP3 | Window 3 | Linear Regression | 17 | 0.117081 | 0.198108 | 0.895145 |
| TP3 | Window 3 | Decision Tree | 17 | 0.064817 | 0.155132 | 0.935704 |
| TP3 | Window 3 | HistGradientBoosting | 17 | 0.062247 | 0.147844 | 0.941603 |
| TP3 | Window 3 | Linear Regression | 42 | 0.117081 | 0.198108 | 0.895145 |
| TP3 | Window 3 | Decision Tree | 42 | 0.064817 | 0.155132 | 0.935704 |
| TP3 | Window 3 | HistGradientBoosting | 42 | 0.062048 | 0.146671 | 0.942526 |
| TP3 | Window 3 | Linear Regression | 73 | 0.117081 | 0.198108 | 0.895145 |
| TP3 | Window 3 | Decision Tree | 73 | 0.064817 | 0.155132 | 0.935704 |
| TP3 | Window 3 | HistGradientBoosting | 73 | 0.062280 | 0.146499 | 0.942660 |
| Motor_current | Window 1 | Persistence | Baseline | 0.786221 | 1.506619 | 0.442768 |
| Motor_current | Window 1 | Linear Regression | 17 | 0.764882 | 1.204497 | 0.643844 |
| Motor_current | Window 1 | Decision Tree | 17 | 0.361169 | 0.824046 | 0.833302 |
| Motor_current | Window 1 | HistGradientBoosting | 17 | 0.283387 | 0.636605 | 0.900512 |
| Motor_current | Window 1 | Linear Regression | 42 | 0.764882 | 1.204497 | 0.643844 |
| Motor_current | Window 1 | Decision Tree | 42 | 0.361979 | 0.824364 | 0.833173 |
| Motor_current | Window 1 | HistGradientBoosting | 42 | 0.295104 | 0.646429 | 0.897418 |
| Motor_current | Window 1 | Linear Regression | 73 | 0.764882 | 1.204497 | 0.643844 |
| Motor_current | Window 1 | Decision Tree | 73 | 0.361169 | 0.824046 | 0.833302 |
| Motor_current | Window 1 | HistGradientBoosting | 73 | 0.289729 | 0.642874 | 0.898543 |
| Motor_current | Window 2 | Persistence | Baseline | 0.811665 | 1.524609 | 0.401696 |
| Motor_current | Window 2 | Linear Regression | 17 | 0.751700 | 1.195192 | 0.632311 |
| Motor_current | Window 2 | Decision Tree | 17 | 0.191594 | 0.492378 | 0.937598 |
| Motor_current | Window 2 | HistGradientBoosting | 17 | 0.206933 | 0.484587 | 0.939557 |
| Motor_current | Window 2 | Linear Regression | 42 | 0.751700 | 1.195192 | 0.632311 |
| Motor_current | Window 2 | Decision Tree | 42 | 0.191594 | 0.492378 | 0.937598 |
| Motor_current | Window 2 | HistGradientBoosting | 42 | 0.204889 | 0.483646 | 0.939791 |
| Motor_current | Window 2 | Linear Regression | 73 | 0.751700 | 1.195192 | 0.632311 |
| Motor_current | Window 2 | Decision Tree | 73 | 0.191594 | 0.492378 | 0.937598 |
| Motor_current | Window 2 | HistGradientBoosting | 73 | 0.204509 | 0.483535 | 0.939819 |
| Motor_current | Window 3 | Persistence | Baseline | 0.753392 | 1.467363 | 0.494590 |
| Motor_current | Window 3 | Linear Regression | 17 | 0.773200 | 1.169370 | 0.679024 |
| Motor_current | Window 3 | Decision Tree | 17 | 0.243916 | 0.638350 | 0.904350 |
| Motor_current | Window 3 | HistGradientBoosting | 17 | 0.243191 | 0.601065 | 0.915197 |
| Motor_current | Window 3 | Linear Regression | 42 | 0.773200 | 1.169370 | 0.679024 |
| Motor_current | Window 3 | Decision Tree | 42 | 0.243916 | 0.638350 | 0.904350 |
| Motor_current | Window 3 | HistGradientBoosting | 42 | 0.246507 | 0.611180 | 0.912319 |
| Motor_current | Window 3 | Linear Regression | 73 | 0.773200 | 1.169370 | 0.679024 |
| Motor_current | Window 3 | Decision Tree | 73 | 0.243916 | 0.638350 | 0.904350 |
| Motor_current | Window 3 | HistGradientBoosting | 73 | 0.246835 | 0.605844 | 0.913843 |

## Interpretation

Mean performance summarizes results across temporal windows and seeds. Standard deviation indicates variation across these experiments.

Positive RMSE reduction indicates lower average RMSE than persistence. Negative reduction indicates worse performance than persistence.

## Limitations

- Repeated seeds do not replace validation on independent datasets.
- Temporal windows belong to the same MetroPT recording.
- Model stability should be assessed alongside changes in operating conditions.
- Results do not establish causal relationships or failure prediction capability.
