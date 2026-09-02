import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from model import HousePriceModel


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Total Bsmt SF",
    "1st Flr SF",
    "Full Bath",
    "TotRms AbvGrd",
    "Year Built",
    "Garage Area",
    "Lot Area"
]

TARGET = "SalePrice"


# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv("data/AmesHousing.csv")

X = data[FEATURES].copy()
y = data[TARGET].copy()

X = X.fillna(X.median())


# =========================================================
# SAME 80/20 TEST SPLIT
# =========================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# LOAD SCALERS
# =========================================================

with open("models/feature_scaler.pkl", "rb") as f:
    feature_scaler = pickle.load(f)

with open("models/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)


# =========================================================
# SCALE TEST DATA
# =========================================================

X_test_scaled = feature_scaler.transform(X_test)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)


# =========================================================
# LOAD MODEL
# =========================================================

model = HousePriceModel()

model.load_state_dict(
    torch.load(
        "models/house_price_model.pth",
        weights_only=True
    )
)

model.eval()


# =========================================================
# PREDICTIONS
# =========================================================

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).numpy()


# Convert predictions back to dollars

predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual = y_test.values


# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    actual,
    predictions
)

mse = mean_squared_error(
    actual,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual,
    predictions
)


# =========================================================
# RESULTS
# =========================================================

print("\n=======================================================")
print("FINAL TEST SET EVALUATION")
print("=======================================================")

print(f"Test samples:     {len(X_test)}")
print(f"MAE:              ${mae:,.2f}")
print(f"MSE:              {mse:,.2f}")
print(f"RMSE:             ${rmse:,.2f}")
print(f"R² Score:         {r2:.4f}")

print("=======================================================")


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predictions,
    alpha=0.6
)

# Perfect prediction line

minimum = min(actual.min(), predictions.min())
maximum = max(actual.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png"
)

plt.close()

print("\nGraph saved:")
print("results/actual_vs_predicted.png")