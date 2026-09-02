import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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
# SAME TEST SPLIT
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
# PREDICT
# =========================================================

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).numpy()


predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual = y_test.values


# =========================================================
# RESIDUALS
# =========================================================

residuals = actual - predictions


print("\n=======================================================")
print("RESIDUAL ANALYSIS")
print("=======================================================")

print(f"Mean Residual: ${np.mean(residuals):,.2f}")
print(f"Mean Absolute Residual: ${np.mean(np.abs(residuals)):,.2f}")
print(f"Residual Std Dev: ${np.std(residuals):,.2f}")


# =========================================================
# RESIDUAL PLOT
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residual (Actual - Predicted)")

plt.title("Residual Plot")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/residual_plot.png"
)

plt.close()

print("\nGraph saved:")
print("results/residual_plot.png")