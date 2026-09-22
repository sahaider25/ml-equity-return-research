import os
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from universe import SP100_SYMBOLS


load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_stock_data(symbols, start_date, end_date):
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment=Adjustment.ALL
    )

    bars = client.get_stock_bars(request)

    return bars.df


def clean_stock_data(df):
    # Convert Alpaca's MultiIndex into regular columns
    df = df.reset_index()

    # Keep only the columns needed for our research
    df = df[
        [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].copy()

    # Convert timestamp into a simple trading date
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    df = df.drop(columns=["timestamp"])

    # Sort observations chronologically
    df = df.sort_values(["date", "symbol"])

    return df.reset_index(drop=True)


if __name__ == "__main__":

    print(f"Downloading data for {len(SP100_SYMBOLS)} stocks...")

    raw_df = get_stock_data(
        symbols=SP100_SYMBOLS,
        start_date="2016-01-01",
        end_date="2026-01-01"
    )

    df = clean_stock_data(raw_df)

    print(df.head())
    print()
    print(f"Rows: {len(df):,}")
    print(f"Stocks: {df['symbol'].nunique()}")
    print(f"Start: {df['date'].min()}")
    print(f"End: {df['date'].max()}")

    df.to_parquet(
        "data/raw/sp100_daily.parquet",
        index=False
    )

    print("\nSaved to data/raw/sp100_daily.parquet")