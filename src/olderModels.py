import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_parquet(
    "data/processed/features.parquet"
)

# Convert dates to pandas datetime
df["date"] = pd.to_datetime(df["date"])

# Remove observations where features or target are unavailable
model_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

print("Usable observations:", len(model_df))


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# Train using data before 2023
train = model_df[
    model_df["date"] < "2023-01-01"
].copy()

# Test using data from 2023 onward
test = model_df[
    model_df["date"] >= "2023-01-01"
].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("Train observations:", len(train))
print("Test observations:", len(test))


# ============================================================
# LINEAR REGRESSION BASELINE
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# ============================================================
# RANDOM FOREST
# ============================================================

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

rf_predictions = rf_model.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(test_df, predictions, name):

    results = test_df[
        ["date", "symbol", TARGET]
    ].copy()

    results["prediction"] = predictions

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            results[TARGET],
            results["prediction"]
        )
    )

    # --------------------------------------------------------
    # DAILY CROSS-SECTIONAL RANK IC
    # --------------------------------------------------------
    #
    # For every trading day, calculate the Spearman correlation
    # between predicted rankings and actual future rankings.
    #
    # Positive IC means stocks predicted to outperform tended
    # to actually outperform.
    # --------------------------------------------------------

    daily_rank_ic = (
        results
        .groupby("date")[["prediction", TARGET]]
        .apply(
            lambda x: x["prediction"].corr(
                x[TARGET],
                method="spearman"
            )
        )
    )

    print()
    print(name)
    print("-" * 40)

    print(f"RMSE: {rmse:.6f}")
    print(
        f"Mean daily rank IC: "
        f"{daily_rank_ic.mean():.4f}"
    )

    return results, daily_rank_ic


# ============================================================
# EVALUATE BOTH MODELS
# ============================================================

linear_results, linear_ic = evaluate_model(
    test,
    linear_predictions,
    "Linear Regression"
)

rf_results, rf_ic = evaluate_model(
    test,
    rf_predictions,
    "Random Forest"
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

print()
print("Random Forest Feature Importance:")
print(importance)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

# Save the Random Forest predictions because we'll use them
# during the portfolio backtest.

rf_results.to_parquet(
    "data/processed/rf_predictions.parquet",
    index=False
)

print()
print(
    "Saved Random Forest predictions to "
    "data/processed/rf_predictions.parquet"
)