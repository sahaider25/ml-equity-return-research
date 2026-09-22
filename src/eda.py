import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data
df = pd.read_parquet("data/raw/sp100_daily.parquet")

print("\n=== DATASET ===")
print("Shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Stocks:", df["symbol"].nunique())


# Missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# Duplicates
duplicates = df.duplicated(subset=["date", "symbol"]).sum()

print("\n=== DUPLICATES ===")
print("Duplicate date-symbol rows:", duplicates)


# Stock history lengths
stock_history = (
    df.groupby("symbol")
      .agg(
          first_date=("date", "min"),
          last_date=("date", "max"),
          observations=("date", "count")
      )
      .sort_values("observations")
)

print("\n=== SHORTEST STOCK HISTORIES ===")
print(stock_history.head(15))


# Number of stocks available each day
stocks_per_day = df.groupby("date")["symbol"].nunique()

print("\n=== STOCKS PER DAY ===")
print(stocks_per_day.describe())


# Daily returns
df = df.sort_values(["symbol", "date"])

df["return_1d"] = (
    df.groupby("symbol")["close"]
      .pct_change(fill_method=None)
)


# Return distribution
print("\n=== DAILY RETURN DISTRIBUTION ===")

print(
    df["return_1d"].describe(
        percentiles=[
            0.001,
            0.01,
            0.05,
            0.5,
            0.95,
            0.99,
            0.999
        ]
    )
)


# Extreme returns
extreme_returns = df.loc[
    df["return_1d"].abs() > 0.25,
    ["date", "symbol", "close", "return_1d"]
].sort_values("return_1d")

print("\n=== EXTREME RETURNS (>25%) ===")
print(extreme_returns)


# Basic data validity
print("\n=== DATA VALIDITY ===")

print("Non-positive closes:", (df["close"] <= 0).sum())
print("Negative volumes:", (df["volume"] < 0).sum())
print("High below low:", (df["high"] < df["low"]).sum())

print(
    "Open outside high-low:",
    ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
)

print(
    "Close outside high-low:",
    ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
)