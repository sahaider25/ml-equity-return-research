import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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

TEST_START = pd.Timestamp("2023-01-01")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    "data/processed/features.parquet"
)

df["date"] = pd.to_datetime(df["date"])

model_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

# Remove any infinite values
model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = model_df.dropna(
    subset=FEATURES + [TARGET]
)


# ============================================================
# PURGED TRAIN / TEST SPLIT
# ============================================================

# Identify the final 10 trading dates before the test period.
# These observations have targets extending into the test period,
# so we exclude them from training.

pre_test_dates = (
    model_df.loc[
        model_df["date"] < TEST_START,
        "date"
    ]
    .drop_duplicates()
    .sort_values()
)

purged_dates = set(
    pre_test_dates.tail(10)
)

train = model_df[
    (model_df["date"] < TEST_START)
    & (~model_df["date"].isin(purged_dates))
].copy()

test = model_df[
    model_df["date"] >= TEST_START
].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("Usable observations:", len(model_df))
print("Train observations:", len(train))
print("Test observations:", len(test))
print("Purged trading days:", len(purged_dates))


# ============================================================
# LINEAR REGRESSION
# ============================================================

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

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
# EVALUATION
# ============================================================

def evaluate_model(test_df, predictions, name):

    results = test_df[
        [
            "date",
            "symbol",
            TARGET,
            "future_return_10d"
        ]
    ].copy()

    results["prediction"] = predictions

    rmse = np.sqrt(
        mean_squared_error(
            results[TARGET],
            results["prediction"]
        )
    )

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

    return results


linear_results = evaluate_model(
    test,
    linear_predictions,
    "Linear Regression"
)

rf_results = evaluate_model(
    test,
    rf_predictions,
    "Random Forest"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print()
print("Random Forest Feature Importance:")
print(importance)


# ============================================================
# SAVE PREDICTIONS
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
print("Saved model predictions.")