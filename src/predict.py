import torch
import pickle
import numpy as np

from model import HousePriceModel


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
# LOAD SCALERS
# =========================================================

with open("models/feature_scaler.pkl", "rb") as f:
    feature_scaler = pickle.load(f)

with open("models/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)


# =========================================================
# USER INPUT
# =========================================================

print("\n==============================================")
print("       HOUSE PRICE PREDICTION SYSTEM")
print("==============================================")

print("\nEnter the following house details:\n")


values = []

for feature in FEATURES:

    value = float(
        input(f"{feature}: ")
    )

    values.append(value)


# =========================================================
# PREPARE INPUT
# =========================================================

input_data = np.array(values).reshape(1, -1)

input_scaled = feature_scaler.transform(
    input_data
)

input_tensor = torch.tensor(
    input_scaled,
    dtype=torch.float32
)


# =========================================================
# PREDICT
# =========================================================

with torch.no_grad():

    prediction_scaled = model(
        input_tensor
    ).numpy()


prediction = target_scaler.inverse_transform(
    prediction_scaled
)[0][0]


# =========================================================
# RESULT
# =========================================================

print("\n==============================================")
print("             PREDICTION RESULT")
print("==============================================")

print(
    f"\nPredicted House Price: ${prediction:,.2f}"
)

print("\n==============================================")