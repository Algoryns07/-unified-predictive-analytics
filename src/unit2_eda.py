"""Unit II: exploratory data analysis and feature selection."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

def run(output_dir="outputs/figures"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data, bunch.target
    print(X.describe().T.head(10))
    print("\nCorrelation with target:\n", X.corrwith(y).sort_values(ascending=False).head(10))
    print("\nIQR outlier counts (top 10):")
    q1, q3 = X.quantile(.25), X.quantile(.75)
    counts = ((X < q1 - 1.5 * (q3-q1)) | (X > q3 + 1.5 * (q3-q1))).sum()
    print(counts.sort_values(ascending=False).head(10))
    selected = SelectKBest(score_func=f_classif, k=10).fit(X, y)
    print("\nTop selected features:", list(X.columns[selected.get_support()]))
    corr = X.corr()
    plt.figure(figsize=(12, 9))
    plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out / "correlation_matrix.png", dpi=150)
    plt.close()
    X.iloc[:, :6].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(out / "feature_distributions.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run()
