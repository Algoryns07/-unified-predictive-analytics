"""Unit VI: CV, GridSearchCV, RandomizedSearchCV, persistence, XGBoost and explainability."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from scipy.stats import loguniform

def run(random_state=42):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=random_state)
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, {"model__C": [.01, .1, 1, 10]}, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("GridSearch best params:", grid.best_params_, "CV F1:", grid.best_score_)
    random_search = RandomizedSearchCV(pipe, {"model__C": loguniform(1e-3, 1e2)},
        n_iter=12, cv=cv, scoring="f1", random_state=random_state, n_jobs=-1)
    random_search.fit(X_train, y_train)
    print("RandomizedSearch best params:", random_search.best_params_)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, "models/best_logistic_pipeline.joblib")
    loaded = joblib.load("models/best_logistic_pipeline.joblib")
    print("Persisted model test score:", loaded.score(X_test, y_test))
    explanation = permutation_importance(loaded, X_test, y_test, n_repeats=5, random_state=random_state, scoring="f1")
    ranking = sorted(zip(X.columns, explanation.importances_mean), key=lambda x: x[1], reverse=True)[:10]
    print("Top permutation features:", ranking)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.05,
                            subsample=.9, colsample_bytree=.9, eval_metric="logloss",
                            random_state=random_state)
        xgb.fit(X_train, y_train)
        print("XGBoost test accuracy:", xgb.score(X_test, y_test))
    except ImportError:
        print("XGBoost not installed; optional section skipped.")

if __name__ == "__main__":
    run()
