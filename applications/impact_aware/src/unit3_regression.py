"""Unit III: regression algorithms and MAE/MSE/RMSE/R2 evaluation."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run(random_state=42):
    bunch = load_diabetes(as_frame=True)
    X, y = bunch.data, bunch.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=random_state)
    models = {
        "Simple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Multiple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Regression": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=.1, max_iter=10000)),
        "Elastic Net": make_pipeline(StandardScaler(), ElasticNet(alpha=.1, l1_ratio=.5, max_iter=10000)),
    }
    # Simple linear regression uses one feature; multiple linear uses all features.
    rows = []
    for name, model in models.items():
        if name == "Simple Linear Regression":
            model.fit(X_train[["bmi"]], y_train)
            pred = model.predict(X_test[["bmi"]])
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rows.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                     "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_test, pred)})
    result = pd.DataFrame(rows).sort_values("RMSE")
    print(result.to_string(index=False))
    return result

if __name__ == "__main__":
    run()
