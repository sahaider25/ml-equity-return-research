import pandas as pd
import numpy as np


HOLDING_PERIOD = 10

# 10 basis points
TRANSACTION_COST = 0.001


def run_backtest(filepath, model_name):

    df = pd.read_parquet(filepath)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    dates = sorted(
        df["date"].unique()
    )

    rebalance_dates = dates[
        ::HOLDING_PERIOD
    ]

    results = []

    for date in rebalance_dates:

        day = df[
            df["date"] == date
        ].copy()

        if len(day) < 20:
            continue

        # Cross-sectional percentile ranking
        day["rank"] = (
            day["prediction"]
            .rank(pct=True)
        )

        longs = day[
            day["rank"] >= 0.90
        ]

        shorts = day[
            day["rank"] <= 0.10
        ]

        if (
            len(longs) == 0
            or len(shorts) == 0
        ):
            continue

        long_return = (
            longs["future_return_10d"]
            .mean()
        )

        short_return = (
            shorts["future_return_10d"]
            .mean()
        )

        gross_return = (
            long_return - short_return
        )

        # Entry + exit for both long and short
        total_cost = (
            4 * TRANSACTION_COST
        )

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

    results["year"] = (
        results["date"].dt.year
    )

    return results


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def performance_stats(
    returns,
    periods_per_year
):

    if len(returns) < 2:
        return None

    cumulative = (
        1 + returns
    ).cumprod()

    annualized_return = (
        cumulative.iloc[-1]
        ** (
            periods_per_year
            / len(returns)
        )
        - 1
    )

    annualized_volatility = (
        returns.std()
        * np.sqrt(periods_per_year)
    )

    if returns.std() != 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(periods_per_year)
        )

    else:
        sharpe = np.nan

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    win_rate = (
        returns > 0
    ).mean()

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }


# ============================================================
# REPORT RESULTS
# ============================================================

def report_backtest(
    results,
    model_name
):

    periods_per_year = (
        252 / HOLDING_PERIOD
    )

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    gross_stats = performance_stats(
        results["gross_return"],
        periods_per_year
    )

    net_stats = performance_stats(
        results["net_return"],
        periods_per_year
    )

    print("\nOVERALL PERFORMANCE")
    print("-" * 60)

    print(
        f"Gross annualized return: "
        f"{gross_stats['annualized_return']:.2%}"
    )

    print(
        f"Net annualized return:   "
        f"{net_stats['annualized_return']:.2%}"
    )

    print(
        f"Net annualized vol:      "
        f"{net_stats['annualized_volatility']:.2%}"
    )

    print(
        f"Net Sharpe:              "
        f"{net_stats['sharpe']:.2f}"
    )

    print(
        f"Net max drawdown:        "
        f"{net_stats['max_drawdown']:.2%}"
    )

    print(
        f"Net win rate:            "
        f"{net_stats['win_rate']:.2%}"
    )

    print("\nYEARLY NET PERFORMANCE")
    print("-" * 60)

    for year, group in results.groupby("year"):

        stats = performance_stats(
            group["net_return"],
            periods_per_year
        )

        print(
            f"{year}: "
            f"Return {stats['annualized_return']:>8.2%} | "
            f"Sharpe {stats['sharpe']:>6.2f} | "
            f"Win Rate {stats['win_rate']:>6.2%}"
        )


# ============================================================
# RUN BACKTESTS
# ============================================================

linear_results = run_backtest(
    "data/processed/linear_predictions.parquet",
    "Linear Regression"
)

rf_results = run_backtest(
    "data/processed/rf_predictions.parquet",
    "Random Forest"
)

report_backtest(
    linear_results,
    "LINEAR REGRESSION"
)

report_backtest(
    rf_results,
    "RANDOM FOREST"
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
print("Saved final backtest results.")