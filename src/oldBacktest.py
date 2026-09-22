import pandas as pd
import numpy as np


HOLDING_PERIOD = 10
TRANSACTION_COST = 0.001  # 10 basis points per side


def run_backtest(filepath, model_name):

    df = pd.read_parquet(filepath)

    df["date"] = pd.to_datetime(df["date"])

    dates = sorted(
        df["date"].unique()
    )

    # Rebalance every 10 trading days
    rebalance_dates = dates[::HOLDING_PERIOD]

    results = []

    for date in rebalance_dates:

        day = df[
            df["date"] == date
        ].copy()

        if len(day) < 20:
            continue

        # Rank stocks by model prediction
        day["rank"] = day["prediction"].rank(
            pct=True
        )

        longs = day[
            day["rank"] >= 0.90
        ]

        shorts = day[
            day["rank"] <= 0.10
        ]

        if len(longs) == 0 or len(shorts) == 0:
            continue

        # Equal-weight long and short portfolios
        long_return = longs[
            "future_return_10d"
        ].mean()

        short_return = shorts[
            "future_return_10d"
        ].mean()

        # Long-short gross return
        gross_return = (
            long_return - short_return
        )

        # Approximate cost of entering/exiting
        # both long and short books
        total_cost = 4 * TRANSACTION_COST

        net_return = (
            gross_return - total_cost
        )

        results.append({
            "date": date,
            "long_return": long_return,
            "short_return": short_return,
            "gross_return": gross_return,
            "net_return": net_return,
            "num_longs": len(longs),
            "num_shorts": len(shorts)
        })

    results = pd.DataFrame(results)

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    periods_per_year = 252 / HOLDING_PERIOD

    mean_return = results["net_return"].mean()

    volatility = results["net_return"].std()

    annualized_return = (
        (1 + results["net_return"]).prod()
        ** (periods_per_year / len(results))
        - 1
    )

    annualized_volatility = (
        volatility * np.sqrt(periods_per_year)
    )

    sharpe = (
        mean_return / volatility
        * np.sqrt(periods_per_year)
    )

    results["equity"] = (
        1 + results["net_return"]
    ).cumprod()

    running_max = (
        results["equity"].cummax()
    )

    results["drawdown"] = (
        results["equity"] / running_max - 1
    )

    max_drawdown = (
        results["drawdown"].min()
    )

    win_rate = (
        results["net_return"] > 0
    ).mean()

    print()
    print(model_name)
    print("=" * 45)

    print(
        f"Backtest periods: {len(results)}"
    )

    print(
        f"Annualized return: "
        f"{annualized_return:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{annualized_volatility:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{sharpe:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Win rate: "
        f"{win_rate:.2%}"
    )

    return results


# ============================================================
# RUN BOTH MODELS
# ============================================================

linear_results = run_backtest(
    "data/processed/linear_predictions.parquet",
    "Linear Regression"
)

rf_results = run_backtest(
    "data/processed/rf_predictions.parquet",
    "Random Forest"
)


# ============================================================
# SAVE RESULTS
# ============================================================

linear_results.to_parquet(
    "data/processed/linear_backtest.parquet",
    index=False
)

rf_results.to_parquet(
    "data/processed/rf_backtest.parquet",
    index=False
)

print()
print("Saved backtest results.")