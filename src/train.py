import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# 5. Create validation set
# -------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42
)


# -------------------------
# 6. Normalize input features
# -------------------------

feature_scaler = StandardScaler()

X_train = feature_scaler.fit_transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)


# -------------------------
# 7. Normalize target
# -------------------------

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
)

y_val_scaled = target_scaler.transform(
    y_val.values.reshape(-1, 1)
)


# -------------------------
# 8. Convert to tensors
# -------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val_scaled,
    dtype=torch.float32
)


# -------------------------
# 9. Create model
# -------------------------

model = HousePriceModel()


# -------------------------
# 10. Loss and optimizer
# -------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# 11. Training
# -------------------------

epochs = 200

train_losses = []
val_losses = []


for epoch in range(epochs):

    # Training mode
    model.train()

    predictions = model(X_train)

    loss = criterion(
        predictions,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    with torch.no_grad():

        val_predictions = model(X_val)

        val_loss = criterion(
            val_predictions,
            y_val
        )


    # Save losses
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())


    # Print progress
    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f}"
        )


# -------------------------
# 12. Save model
# -------------------------

torch.save(
    model.state_dict(),
    "models/house_price_model.pth"
)


# -------------------------
# 13. Save target scaler
# -------------------------

joblib.dump(
    target_scaler,
    "models/target_scaler.pkl"
)


# -------------------------
# 14. Plot training curves
# -------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_validation_loss.png"
)

plt.show()


print("\nTraining completed!")
print("Model saved successfully!")
print("Target scaler saved successfully!")