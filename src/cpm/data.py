"""Download daily BTC prices (Coinbase) and implied volatility (Deribit DVOL)."""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
START = "2021-04-01"  # Deribit's volatility index only exists from spring 2021


def last_complete_day():
    """Yesterday at 00:00 UTC. Today's candle is still forming, so we never use it."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=1)


def fetch_prices(product="BTC-USD", start=START):
    """Daily open/high/low/close/volume candles from Coinbase."""
    cursor = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end = last_complete_day()
    rows = []
    while cursor <= end:
        # Coinbase returns at most 300 candles per request, so we ask in chunks.
        chunk_end = min(cursor + timedelta(days=299), end)
        response = requests.get(
            COINBASE_URL.format(product=product),
            params={
                "granularity": 86400,  # seconds in one day
                "start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            headers={"User-Agent": "crypto-movement-predictor"},
            timeout=30,
        )
        response.raise_for_status()
        rows.extend(response.json())
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # be polite to the API

    # Coinbase's column order is: time, low, high, open, close, volume
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    df = df[df.index <= end.replace(tzinfo=None)]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_implied_vol(currency="BTC", start=START):
    """Daily closing value of Deribit's implied volatility index (DVOL)."""
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    rows = []
    while True:
        response = requests.get(
            DERIBIT_URL,
            params={
                "currency": currency,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": "1D",
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["result"]
        rows.extend(result["data"])
        # Deribit sends the newest block first, plus a "continuation" marker
        # telling us where to ask for the next (older) block.
        if not result["data"] or not result.get("continuation"):
            break
        end_ms = result["continuation"]
        time.sleep(0.3)

    df = pd.DataFrame(rows, columns=["time", "iv_open", "iv_high", "iv_low", "iv_close"])
    df["date"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    df = df[df.index <= last_complete_day().replace(tzinfo=None)]
    return df[["iv_close"]].rename(columns={"iv_close": "implied_vol"})


def main():
    DATA_DIR.mkdir(exist_ok=True)
    prices = fetch_prices()
    implied_vol = fetch_implied_vol()
    # Keep only days where we have both a price and an implied volatility reading.
    df = prices.join(implied_vol, how="inner")
    path = DATA_DIR / "btc_daily.csv"
    df.to_csv(path)
    print(f"Saved {len(df)} days to {path}")
    print(f"From {df.index.min().date()} to {df.index.max().date()}")
    print(df.tail())


if __name__ == "__main__":
    main()