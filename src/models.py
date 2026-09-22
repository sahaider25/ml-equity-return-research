import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "volume_ratio",
    "price_to_ma20",
    "relative_momentum_20d"
]

TARGET = "target_10d"

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    "data/processed/features.parquet"
)

df["date"] = pd.to_datetime(df["date"])

model_df = df.dropna(
    subset=FEATURES + [TARGET, "future_return_10d"]
).copy()

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = model_df.dropna(
    subset=FEATURES + [TARGET]
)


# ============================================================
# CLIP EXTREME FEATURE VALUES
# ============================================================
#
# Extreme observations can create numerical instability.
# We winsorize each feature at the 0.1% and 99.9% levels.
#
# IMPORTANT:
# Bounds are calculated from training data only during each
# walk-forward iteration.
# ============================================================

def clip_features(train, test):

    train = train.copy()
    test = test.copy()

    for feature in FEATURES:

        lower = train[feature].quantile(0.001)
        upper = train[feature].quantile(0.999)

        train[feature] = train[feature].clip(
            lower,
            upper
        )

        test[feature] = test[feature].clip(
            lower,
            upper
        )

    return train, test


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_rank_ic(results):

    return (
        results
        .groupby("date")[["prediction", TARGET]]
        .apply(
            lambda x: x["prediction"].corr(
                x[TARGET],
                method="spearman"
            )
        )
    )


# ============================================================
# WALK-FORWARD TESTING
# ============================================================

all_linear_results = []
all_rf_results = []

print("\nWALK-FORWARD MODELING")
print("=" * 55)


for year in TEST_YEARS:

    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year + 1}-01-01"
    )

    # --------------------------------------------------------
    # Find final 10 trading days before test year
    # --------------------------------------------------------

    pre_test_dates = (
        model_df.loc[
            model_df["date"] < test_start,
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    purged_dates = set(
        pre_test_dates.tail(10)
    )

    # --------------------------------------------------------
    # Expanding training window
    # --------------------------------------------------------

    train = model_df[
        (model_df["date"] < test_start)
        & (~model_df["date"].isin(purged_dates))
    ].copy()

    test = model_df[
        (model_df["date"] >= test_start)
        & (model_df["date"] < test_end)
    ].copy()

    train, test = clip_features(
        train,
        test
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    linear_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        Ridge(alpha=1.0)
    )
    ])

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = (
        linear_model.predict(X_test)
    )

    linear_results = test[
        [
            "date",
            "symbol",
            TARGET,
            "future_return_10d"
        ]
    ].copy()

    linear_results["prediction"] = (
        linear_predictions
    )

    linear_results["test_year"] = year

    all_linear_results.append(
        linear_results
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_predictions = (
        rf_model.predict(X_test)
    )

    rf_results = test[
        [
            "date",
            "symbol",
            TARGET,
            "future_return_10d"
        ]
    ].copy()

    rf_results["prediction"] = (
        rf_predictions
    )

    rf_results["test_year"] = year

    all_rf_results.append(
        rf_results
    )

    # --------------------------------------------------------
    # YEARLY IC
    # --------------------------------------------------------

    linear_ic = calculate_rank_ic(
        linear_results
    ).mean()

    rf_ic = calculate_rank_ic(
        rf_results
    ).mean()

    print()
    print(f"Test Year: {year}")
    print("-" * 55)

    print(
        f"Training observations: {len(train):,}"
    )

    print(
        f"Test observations:     {len(test):,}"
    )

    print(
        f"Ridge Rank IC:        {linear_ic:.4f}"
    )

    print(
        f"Random Forest Rank IC: {rf_ic:.4f}"
    )


# ============================================================
# COMBINE ALL OUT-OF-SAMPLE PREDICTIONS
# ============================================================

linear_results = pd.concat(
    all_linear_results,
    ignore_index=True
)

rf_results = pd.concat(
    all_rf_results,
    ignore_index=True
)


# ============================================================
# OVERALL WALK-FORWARD RESULTS
# ============================================================

linear_ic = calculate_rank_ic(
    linear_results
)

rf_ic = calculate_rank_ic(
    rf_results
)

print()
print("=" * 55)
print("OVERALL WALK-FORWARD RESULTS")
print("=" * 55)

print(
    f"Ridge Regression Mean Rank IC: "
    f"{linear_ic.mean():.4f}"
)

print(
    f"Random Forest Mean Rank IC:     "
    f"{rf_ic.mean():.4f}"
)


# ============================================================
# SAVE OUT-OF-SAMPLE PREDICTIONS
# ============================================================

linear_results.to_parquet(
    "data/processed/linear_predictions.parquet",
    index=False
)

rf_results.to_parquet(
    "data/processed/rf_predictions.parquet",
    index=False
)

print()
print(
    "Saved walk-forward predictions."
)