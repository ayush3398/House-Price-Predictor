import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import HousePriceModel


# Load dataset
df = pd.read_csv("data/AmesHousing.csv")


# Features
features = [
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

target = "SalePrice"


# X and y
X = df[features].copy()
y = df[target].copy()

# Handle missing values
X = X.fillna(X.median())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Normalize input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert test data to tensor
X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)


# Load model
model = HousePriceModel()

model.load_state_dict(
    torch.load(
        "models/house_price_model.pth",
        weights_only=True
    )
)

model.eval()


# Load target scaler
target_scaler = joblib.load(
    "models/target_scaler.pkl"
)


# Make predictions
with torch.no_grad():
    predictions = model(X_test)


predictions = predictions.numpy()

# Convert back to dollars
predictions = target_scaler.inverse_transform(
    predictions
).flatten()


# Calculate residuals
residuals = y_test.values - predictions


# Plot residuals
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

plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Prediction Error / Residual Plot")

plt.tight_layout()

plt.savefig(
    "results/residual_plot.png"
)

plt.show()