import pytest

from src.run_multiseed import run_multiseed


@pytest.mark.parametrize(
    "seeds",
    [
        (-1,),
        (1, -5),
        (1.5,),
        (True,),
        ("42",),
    ],
)
def test_multiseed_rejects_invalid_seed_values(seeds, tmp_path):
    with pytest.raises(ValueError):
        run_multiseed(seeds=seeds, output_dir=tmp_path)


def test_multiseed_rejects_empty_seed_collection(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        run_multiseed(seeds=(), output_dir=tmp_path)


def test_multiseed_rejects_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_multiseed(seeds=(11, 42, 11), output_dir=tmp_path)


def test_multiseed_accepts_non_negative_integer_seeds():
    import numpy as np

    from src.run_multiseed import DEFAULT_SEEDS

    assert all(isinstance(seed, int) and seed >= 0 for seed in DEFAULT_SEEDS)
    assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS)
