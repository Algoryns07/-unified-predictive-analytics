"""PSI, KS and Wasserstein drift measurements for numeric features."""
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

def psi(reference, current, bins=10, epsilon=1e-6):
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, bins=edges)[0].astype(float)
    b = np.histogram(cur, bins=edges)[0].astype(float)
    a = np.clip(a / max(a.sum(), 1), epsilon, None)
    b = np.clip(b / max(b.sum(), 1), epsilon, None)
    return float(np.sum((b-a) * np.log(b/a)))

def compare_feature(reference, current):
    ks = ks_2samp(reference, current)
    return {"psi": psi(reference, current), "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "wasserstein": float(wasserstein_distance(reference, current))}

def compare_frames(reference, current, psi_threshold=.10, ks_alpha=.05, wasserstein_threshold=.20):
    result = {}
    for col in reference.columns.intersection(current.columns):
        stats = compare_feature(reference[col], current[col])
        stats["drift"] = bool(stats["psi"] >= psi_threshold or stats["ks_p_value"] < ks_alpha
                              or stats["wasserstein"] >= wasserstein_threshold)
        result[col] = stats
    return {"features": result, "drifted_features": [k for k,v in result.items() if v["drift"]],
            "any_drift": any(v["drift"] for v in result.values())}
