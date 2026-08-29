import joblib
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import HousePriceModel


# -------------------------
# 1. Load dataset
# -------------------------

df = pd.read_csv("data/AmesHousing.csv")


# -------------------------
# 2. Select features
# -------------------------

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


X = df[features].copy()
y = df[target].copy()


# -------------------------
# 3. Handle missing values
# -------------------------

X = X.fillna(X.median())


# -------------------------
# 4. Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -------------------------
# 5. Normalize
# -------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------
# 6. Convert test data
#    to tensors
# -------------------------

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)


# -------------------------
# 7. Load trained model
# -------------------------

model = HousePriceModel()

model.load_state_dict(
    torch.load(
        "models/house_price_model.pth",
        weights_only=True
    )
)

model.eval()


# -------------------------
# 8. Make predictions
# -------------------------

target_scaler = joblib.load(
    "models/target_scaler.pkl"
)

with torch.no_grad():
    predictions = model(X_test)


predictions = predictions.numpy()

# Convert predictions back to dollars
predictions = target_scaler.inverse_transform(
    predictions
).flatten()


# -------------------------
# 9. Calculate metrics
# -------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nMODEL PERFORMANCE")
print("-------------------------")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")


# -------------------------
# 10. Actual vs Predicted
# -------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
min_price = min(y_test.min(), predictions.min())
max_price = max(y_test.max(), predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price]
)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png"
)

plt.show()