"""Unit I: loading, missing-value handling, encoding, scaling and splitting."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_dataset():
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()

def prepare_data(test_size=0.25, random_state=42):
    X, y = load_dataset()
    # Dataset is numeric; imputation and scaling are still demonstrated.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_ready = preprocessing.fit_transform(X_train)
    X_test_ready = preprocessing.transform(X_test)
    return X_train_ready, X_test_ready, y_train, y_test, preprocessing

if __name__ == "__main__":
    Xtr, Xte, ytr, yte, _ = prepare_data()
    print("Train:", Xtr.shape, "Test:", Xte.shape)
    print("Missing values in source:", int(load_dataset()[0].isna().sum().sum()))
