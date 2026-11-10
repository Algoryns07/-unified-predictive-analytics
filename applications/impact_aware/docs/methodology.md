# Research methodology

## Problem
Feature distributions in deployed systems can change. A drift alarm alone does not establish that predictions have become worse. The project measures feature drift and evaluates its association with predictive performance loss.

## Experimental flow
1. Load and preprocess a labelled reference dataset.
2. Split data into model-training and reference-evaluation subsets.
3. Fit a baseline classifier using training data only.
4. Construct monitoring windows with documented shifts.
5. Compare feature distributions using PSI, KS and Wasserstein distance.
6. Evaluate the fixed baseline model on each labelled monitoring window.
7. Calculate the difference in F1 from the reference evaluation score.
8. Assign an impact class and recommendation using a transparent rule.
9. Cluster monitoring-window summaries to explore recurring drift patterns.
10. Compare the policy with retraining baselines.

## Drift versus impact
Drift is measured on input features. Impact is measured using labels and predictive metrics. The two are related but not equivalent. Monitoring labels must not be used to fit the original model.

## Decision rule
The starter policy labels a window as harmful when F1 degradation crosses a configured threshold. It recommends retraining only when the estimated loss also crosses a cost/benefit hurdle. This is a transparent baseline, not a validated optimal policy.

## Limitations
The starter uses a controlled benchmark dataset and simulated feature shifts. Simulation cannot establish performance on a live production stream. Some shifts can alter class proportions or make the task easier; therefore interpret performance changes rather than assuming every shift causes degradation. Add real datasets, temporal splits where possible, repeated runs and uncertainty intervals before drawing research conclusions.
