import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

ridge_backtest = pd.read_parquet(
    "data/processed/linear_backtest.parquet"
)

rf_backtest = pd.read_parquet(
    "data/processed/rf_backtest.parquet"
)

ridge_predictions = pd.read_parquet(
    "data/processed/linear_predictions.parquet"
)

rf_predictions = pd.read_parquet(
    "data/processed/rf_predictions.parquet"
)

ridge_backtest["date"] = pd.to_datetime(
    ridge_backtest["date"]
)

rf_backtest["date"] = pd.to_datetime(
    rf_backtest["date"]
)


# ============================================================
# 1. CUMULATIVE PORTFOLIO PERFORMANCE
# ============================================================

ridge_backtest["equity"] = (
    1 + ridge_backtest["net_return"]
).cumprod()

rf_backtest["equity"] = (
    1 + rf_backtest["net_return"]
).cumprod()

plt.figure(figsize=(10, 6))

plt.plot(
    ridge_backtest["date"],
    ridge_backtest["equity"],
    label="Ridge Regression"
)

plt.plot(
    rf_backtest["date"],
    rf_backtest["equity"],
    label="Random Forest"
)

plt.axhline(
    1,
    linestyle="--",
    linewidth=1
)

plt.title(
    "Walk-Forward Long-Short Portfolio Performance"
)

plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/equity_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. YEARLY NET RETURNS
# ============================================================

ridge_yearly = (
    ridge_backtest
    .groupby(
        ridge_backtest["date"].dt.year
    )["net_return"]
    .apply(
        lambda x: (1 + x).prod() - 1
    )
)

rf_yearly = (
    rf_backtest
    .groupby(
        rf_backtest["date"].dt.year
    )["net_return"]
    .apply(
        lambda x: (1 + x).prod() - 1
    )
)

yearly = pd.DataFrame({
    "Ridge Regression": ridge_yearly,
    "Random Forest": rf_yearly
})

ax = yearly.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "Out-of-Sample Net Return by Year"
)

ax.set_xlabel("Year")
ax.set_ylabel("Net Return")

plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/yearly_returns.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. YEARLY RANK IC
# ============================================================

def yearly_rank_ic(df):

    daily_ic = (
        df.groupby("date")[
            ["prediction", "target_10d"]
        ]
        .apply(
            lambda x:
            x["prediction"].corr(
                x["target_10d"],
                method="spearman"
            )
        )
    )

    daily_ic.index = pd.to_datetime(
        daily_ic.index
    )

    return daily_ic.groupby(
        daily_ic.index.year
    ).mean()


ridge_ic = yearly_rank_ic(
    ridge_predictions
)

rf_ic = yearly_rank_ic(
    rf_predictions
)

ic_data = pd.DataFrame({
    "Ridge Regression": ridge_ic,
    "Random Forest": rf_ic
})

ax = ic_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "Mean Daily Cross-Sectional Rank IC"
)

ax.set_xlabel("Year")
ax.set_ylabel("Spearman Rank IC")

plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/yearly_rank_ic.png",
    dpi=300
)

plt.close()


print("Saved:")
print("results/equity_curve.png")
print("results/yearly_returns.png")
print("results/yearly_rank_ic.png")