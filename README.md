# Impact-Aware Data Drift Monitoring and Cost-Sensitive Model Retraining

**Course:** CSE570 – Machine Learning with Python (Units I–VI)  
**Project type:** Course-mapped ML implementation + research prototype  
**Primary task:** Binary classification under changing input-data distributions.

## Research problem

A change in incoming feature distributions does not automatically mean that a model has become less useful. This project detects feature-distribution changes, measures their effect on a fixed classifier, distinguishes harmful from non-harmful drift, and recommends whether retraining is justified under a transparent cost threshold.

**Research questions**
1. Does detected feature drift necessarily reduce predictive performance?
2. Can drift windows be grouped into interpretable patterns?
3. Can an impact-aware retraining policy avoid unnecessary retraining while protecting predictive performance?

This repository is a course-mapped prototype. Its synthetic drift experiments validate the workflow; they are not evidence of real-world superiority. Add and document a suitable real dataset before making dissertation claims.

## Explicit mapping to CSE570 Units I–VI

| Unit | Syllabus topics | Project implementation |
|---|---|---|
| I | Python, NumPy, Pandas, loading, missing values, encoding, scaling, train-test split | `src/unit1_preprocessing.py` |
| II | EDA, visualization, correlation, outliers, feature selection/transformation, imbalance, pipelines | `src/unit2_eda.py` |
| III | Simple/Multiple/Polynomial Regression, Ridge, Lasso, Elastic Net, MAE/MSE/RMSE/R² | `src/unit3_regression.py` |
| IV | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, classification metrics | `src/unit4_classification.py` |
| V | K-Means, Hierarchical, DBSCAN, PCA, silhouette score | `src/unit5_clustering_pca.py` |
| VI | Cross-validation, GridSearchCV, RandomizedSearchCV, persistence, pipelines, XGBoost, explainable AI | `src/unit6_optimization.py` |

The syllabus lists Random Forest under **classification**, not regression. This project uses Random Forest Classifier in Unit IV. Unit III regression models are evaluated separately on a regression dataset.

## Repository structure

```text
impact-aware-data-drift-course-mapped/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/default.yaml
├── data/README.md
├── docs/
│   ├── unit_mapping.md
│   ├── methodology.md
│   ├── experiment_protocol.md
│   └── research_log.md
├── src/
│   ├── unit1_preprocessing.py
│   ├── unit2_eda.py
│   ├── unit3_regression.py
│   ├── unit4_classification.py
│   ├── drift_detection.py
│   ├── impact_policy.py
│   ├── unit5_clustering_pca.py
│   ├── unit6_optimization.py
│   └── run_project.py
├── tests/
│   ├── test_drift.py
│   └── test_policy.py
└── outputs/.gitkeep
```

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the complete project

```powershell
python -m src.run_project
```

Run individual units:
```powershell
python -m src.unit1_preprocessing
python -m src.unit2_eda
python -m src.unit3_regression
python -m src.unit4_classification
python -m src.unit5_clustering_pca
python -m src.unit6_optimization
```

Run tests:
```powershell
python -m pytest
```

Outputs are written under `outputs/`. Generated outputs are excluded from Git by default; commit selected figures/reports only when they are useful and reproducible.

## Main drift experiment

The main demonstration uses the scikit-learn breast-cancer binary classification dataset. A baseline Logistic Regression classifier is trained on a stratified training split. A held-out reference set and simulated monitoring windows are compared. The monitoring scenarios introduce documented feature shifts while preserving the original labels for a controlled experiment.

Drift measures:
- PSI (Population Stability Index)
- KS two-sample test
- Wasserstein distance

Impact measures:
- Accuracy, precision, recall, F1, ROC-AUC
- F1 drop relative to the reference evaluation window

Decision classes:
- `NO_DRIFT`
- `DRIFT_NO_HARM`
- `DRIFT_HARMFUL`

Actions:
- `MONITOR`
- `INVESTIGATE`
- `RETRAIN`

The policy is a transparent baseline rule, not an optimally learned economic policy. Tune thresholds on validation windows and freeze them before final testing.

## Important research safeguards

- Feature/data drift is not identical to concept drift.
- A statistical drift alarm does not prove performance harm.
- Labels are required to measure actual predictive impact. With delayed labels, the system can only raise a provisional drift alert.
- Never train the baseline model on monitoring-window labels.
- Do not choose thresholds after inspecting final test results.
- A random split is not a temporal evaluation. For real time-dependent data, split chronologically.
- Report dataset source/version, licence, seed, preprocessing, split, model settings, thresholds and limitations.
- Do not claim the proposed policy is better until fair baseline comparisons support that conclusion.

## Next steps for dissertation-grade work

1. Run all six unit modules and inspect outputs.
2. Add a real, appropriately licensed dataset and document its source.
3. Create chronological or otherwise defensible monitoring windows.
4. Compare all six Unit IV classifiers on the same splits.
5. Add repeated seeds and confidence intervals.
6. Compare always-retrain, drift-only, performance-only and proposed policy.
7. Measure retraining time and number of retraining events.
8. Record results and limitations in `docs/research_log.md`.
