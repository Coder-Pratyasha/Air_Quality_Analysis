import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. Load Train and Test Data
# --------------------------------------------------

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X_train = train.drop(
    columns=["target_pm25", "Datetime"]
)

y_train = train["target_pm25"]

X_test = test.drop(
    columns=["target_pm25", "Datetime"]
)

y_test = test["target_pm25"]


# --------------------------------------------------
# 3. Identify Categorical and Numerical Features
# --------------------------------------------------

categorical_features = ["City"]

numerical_features = [
    column for column in X_train.columns
    if column not in categorical_features
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. Define Models
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 6. Train and Evaluate Models
# --------------------------------------------------

results = []

best_model = None
best_model_name = None
best_rmse = float("inf")


for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Keep best model based on RMSE
    if rmse < best_rmse:

        best_rmse = rmse
        best_model = pipeline
        best_model_name = name


# --------------------------------------------------
# 7. Model Comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="RMSE"
    ).to_string(index=False)
)


# --------------------------------------------------
# 8. Best Model
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print("RMSE:", round(best_rmse, 4))


# --------------------------------------------------
# 9. Save Results
# --------------------------------------------------

results_df.to_csv(
    "data/model_results.csv",
    index=False
)


# --------------------------------------------------
# 10. Save Best Model
# --------------------------------------------------

import joblib

joblib.dump(
    best_model,
    "models/pm25_model.pkl"
)

print("\nModel saved to:")
print("models/pm25_model.pkl")

print("\nResults saved to:")
print("data/model_results.csv")