import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load Test Data
# --------------------------------------------------

test = pd.read_csv("data/test.csv")

test["Datetime"] = pd.to_datetime(test["Datetime"])

print("Test data loaded:", test.shape)


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X_test = test.drop(
    columns=["target_pm25", "Datetime"]
)

y_test = test["target_pm25"]


# --------------------------------------------------
# 3. Load Best Model
# --------------------------------------------------

model = joblib.load(
    "models/pm25_model.pkl"
)

print("XGBoost model loaded successfully.")


# --------------------------------------------------
# 4. Generate Predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 5. Create Prediction DataFrame
# --------------------------------------------------

results = pd.DataFrame({
    "Datetime": test["Datetime"],
    "City": test["City"],
    "Actual_PM25": y_test.values,
    "Predicted_PM25": predictions
})

results["Error"] = (
    results["Actual_PM25"]
    - results["Predicted_PM25"]
)

results["Absolute_Error"] = (
    results["Error"].abs()
)


# --------------------------------------------------
# 6. Display Prediction Examples
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results.head(10).to_string(
        index=False
    )
)


# --------------------------------------------------
# 7. Worst Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("10 LARGEST PREDICTION ERRORS")
print("=" * 60)

worst_predictions = (
    results
    .sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(10)
)

print(
    worst_predictions.to_string(
        index=False
    )
)


# --------------------------------------------------
# 8. Actual vs Predicted Plot
# --------------------------------------------------

# Use a sample because plotting all 224k points
# would make the graph difficult to read.

sample = results.head(1000)

plt.figure(figsize=(14, 6))

plt.plot(
    sample["Datetime"],
    sample["Actual_PM25"],
    label="Actual PM2.5"
)

plt.plot(
    sample["Datetime"],
    sample["Predicted_PM25"],
    label="Predicted PM2.5"
)

plt.title(
    "Actual vs Predicted PM2.5"
)

plt.xlabel("Datetime")

plt.ylabel("PM2.5")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/actual_vs_predicted.png"
)

plt.show()


# --------------------------------------------------
# 9. Prediction Error Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    results["Error"],
    bins=50
)

plt.title(
    "Distribution of Prediction Errors"
)

plt.xlabel(
    "Prediction Error (Actual - Predicted)"
)

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "data/prediction_error.png"
)

plt.show()


# --------------------------------------------------
# 10. Save Predictions
# --------------------------------------------------

results.to_csv(
    "data/predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nPrediction file saved to:")
print("data/predictions.csv")

print("\nActual vs Predicted plot saved to:")
print("data/actual_vs_predicted.png")

print("\nPrediction error plot saved to:")
print("data/prediction_error.png")