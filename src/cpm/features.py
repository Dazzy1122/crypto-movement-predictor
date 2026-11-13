"""Turn raw daily data into model features and the next-day up/down target."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/btc_daily.csv")
DATASET_PATH = Path("data/dataset.csv")
DAYS_PER_YEAR = 365  # crypto trades every day, unlike stocks (252)

FEATURES = [
    "ret_1", "ret_5", "ret_20",                    # momentum
    "rv_7", "rv_30",                               # realized volatility
    "implied_vol", "iv_change_1", "iv_change_5",   # options market
    "vrp", "iv_zscore",                            # implied vs realized
    "range", "volume_z", "rsi_14", "ma_gap",       # classic technicals
]


def load_raw(path=RAW_PATH):
    return pd.read_csv(path, index_col="date", parse_dates=True)


def rsi(close, window=14):
    """Relative Strength Index: 0-100, high means recent days were mostly up."""
    change = close.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = (-change.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(raw):
    """Every value on row t uses only information available at the close of day t."""
    df = pd.DataFrame(index=raw.index)
    log_close = np.log(raw["close"])
    daily_ret = log_close.diff()
    iv = raw["implied_vol"]

    # Momentum: how much price moved over the last 1, 5 and 20 days.
    df["ret_1"] = daily_ret
    df["ret_5"] = log_close.diff(5)
    df["ret_20"] = log_close.diff(20)

    # Realized volatility: how much price actually moved, annualized and in
    # percent so it is on the same scale as the implied volatility index.
    annualize = np.sqrt(DAYS_PER_YEAR) * 100
    df["rv_7"] = daily_ret.rolling(7).std() * annualize
    df["rv_30"] = daily_ret.rolling(30).std() * annualize

    # Implied volatility: how much the options market expects price to move.
    df["implied_vol"] = iv
    df["iv_change_1"] = iv.diff()
    df["iv_change_5"] = iv.diff(5)

    # Volatility risk premium: expected movement minus actual movement.
    df["vrp"] = iv - df["rv_30"]
    # Is implied volatility high or low compared with its own last 60 days?
    df["iv_zscore"] = (iv - iv.rolling(60).mean()) / iv.rolling(60).std()

    # Classic technical indicators.
    df["range"] = (raw["high"] - raw["low"]) / raw["close"]
    log_volume = np.log(raw["volume"])
    df["volume_z"] = (log_volume - log_volume.rolling(30).mean()) / log_volume.rolling(30).std()
    df["rsi_14"] = rsi(raw["close"])
    df["ma_gap"] = raw["close"] / raw["close"].rolling(50).mean() - 1
    return df


def build_dataset(raw):
    """Features plus what we are trying to predict: tomorrow's direction."""
    df = build_features(raw)
    # shift(-1) pulls tomorrow's return onto today's row. This is the ONLY
    # place the future is allowed to appear, and only as the answer key.
    df["next_ret"] = np.log(raw["close"]).diff().shift(-1)
    df["target"] = (df["next_ret"] > 0).astype(int)
    # Drops the warm-up rows (rolling windows not full yet) and the final
    # row (tomorrow has not happened yet, so it has no answer).
    return df.dropna()


def main():
    dataset = build_dataset(load_raw())
    dataset.to_csv(DATASET_PATH)
    print(f"Saved {len(dataset)} rows and {len(FEATURES)} features to {DATASET_PATH}")
    print(f"From {dataset.index.min().date()} to {dataset.index.max().date()}")
    print(f"Share of up days: {dataset['target'].mean():.1%}")
    print(dataset[["ret_1", "rv_30", "implied_vol", "vrp", "next_ret", "target"]].tail().round(3))


if __name__ == "__main__":
    main()