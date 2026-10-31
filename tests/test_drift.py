import numpy as np
import pandas as pd
from src.drift_detection import psi, compare_frames

def test_psi_nonnegative_for_same_distribution():
    x = np.random.default_rng(4).normal(size=1000)
    assert psi(x, x) >= 0

def test_large_shift_detected():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"x": rng.normal(4, 1, 1000)})
    assert compare_frames(ref, cur)["any_drift"]
