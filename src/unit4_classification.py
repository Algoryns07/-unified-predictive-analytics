"""Unit IV: compare all six syllabus classification algorithms."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def run(random_state=42):
    bunch = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        bunch.data, bunch.target, test_size=.25, random_state=random_state, stratify=bunch.target
    )
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=random_state)),
        "Naive Bayes": GaussianNB(),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                     "precision": precision_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred, zero_division=0),
                     "f1": f1_score(y_test, pred, zero_division=0),
                     "roc_auc": roc_auc_score(y_test, prob)})
        if name == "Random Forest":
            print("Random Forest confusion matrix:\n", confusion_matrix(y_test, pred))
    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(result.to_string(index=False))
    return result

if __name__ == "__main__":
    run()
