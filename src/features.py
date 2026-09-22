import pandas as pd
import numpy as np


def build_features(df):

    df = df.copy()
    df = df.sort_values(["symbol", "date"])

    grouped = df.groupby("symbol")

    # -----------------------
    # RETURNS / MOMENTUM
    # -----------------------

    df["return_1d"] = grouped["close"].pct_change(fill_method=None)

    df["momentum_5d"] = grouped["close"].pct_change(
        periods=5,
        fill_method=None
    )

    df["momentum_20d"] = grouped["close"].pct_change(
        periods=20,
        fill_method=None
    )

    df["momentum_60d"] = grouped["close"].pct_change(
        periods=60,
        fill_method=None
    )

    # -----------------------
    # VOLATILITY
    # -----------------------

    df["volatility_20d"] = (
        df.groupby("symbol")["return_1d"]
          .transform(lambda x: x.rolling(20).std())
    )

    # -----------------------
    # VOLUME
    # -----------------------

    df["volume_avg_20d"] = (
        grouped["volume"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["volume_ratio"] = (
        df["volume"] / df["volume_avg_20d"]
    )

    # -----------------------
    # TREND
    # -----------------------

    df["ma_20d"] = (
        grouped["close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["price_to_ma20"] = (
        df["close"] / df["ma_20d"] - 1
    )

    # -----------------------
    # CROSS-SECTIONAL RELATIVE MOMENTUM
    # -----------------------

    daily_median_momentum = (
        df.groupby("date")["momentum_20d"]
          .transform("median")
    )

    df["relative_momentum_20d"] = (
        df["momentum_20d"] - daily_median_momentum
    )

        # -----------------------
    # TARGET
    # -----------------------

    # Future 10-day return
    df["future_return_10d"] = (
        grouped["close"].shift(-10) / df["close"] - 1
    )

    # Median future return across stocks on each date
    median_future_return = (
        df.groupby("date")["future_return_10d"]
          .transform("median")
    )

    # Cross-sectional relative return
    df["target_10d"] = (
        df["future_return_10d"] - median_future_return
    )

    return df

if __name__ == "__main__":

    df = pd.read_parquet(
        "data/raw/sp100_daily.parquet"
    )

    df = build_features(df)

    print(df.head(25))

    print("\nRows:", len(df))

    feature_columns = [
        "momentum_5d",
        "momentum_20d",
        "momentum_60d",
        "volatility_20d",
        "volume_ratio",
        "price_to_ma20",
        "relative_momentum_20d"
    ]

    print("\nMissing feature values:")
    print(df[feature_columns].isnull().sum())

    df.to_parquet(
        "data/processed/features.parquet",
        index=False
    )

    print("\nSaved processed features.")