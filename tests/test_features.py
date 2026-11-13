import numpy as np
import pandas as pd

from cpm.features import FEATURES, build_dataset, build_features


def make_raw(n=300, seed=0):
    """Fake but realistic-looking market data, so tests never need the internet."""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
            "implied_vol": rng.uniform(40, 90, n),
        },
        index=pd.date_range("2022-01-01", periods=n, name="date"),
    )


def test_features_do_not_use_future_data():
    # If we delete the last 100 days, the features for the first 200 days
    # must not change. If they did, they were peeking at the future.
    raw = make_raw()
    full = build_features(raw)
    truncated = build_features(raw.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], truncated)


def test_target_is_next_day_direction():
    raw = make_raw()
    dataset = build_dataset(raw)
    day = dataset.index[10]
    next_day = raw.index[raw.index.get_loc(day) + 1]
    expected = int(raw.loc[next_day, "close"] > raw.loc[day, "close"])
    assert dataset.loc[day, "target"] == expected


def test_dataset_is_clean():
    dataset = build_dataset(make_raw())
    assert not dataset.isna().any().any()
    assert list(dataset.columns) == FEATURES + ["next_ret", "target"]