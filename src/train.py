import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import HousePriceModel


# =========================================================
# SETTINGS
# =========================================================

EPOCHS = 300
LEARNING_RATE = 0.001
PATIENCE = 20
WEIGHT_DECAY = 0.0001
RANDOM_STATE = 42

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

# Fill missing feature values
X = X.fillna(X.median())


# =========================================================
# HOLD OUT FINAL TEST SET
# =========================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\n=======================================================")
print("FINAL MODEL TRAINING")
print("=======================================================")

print(f"Total samples:       {len(X)}")
print(f"Development samples: {len(X_dev)}")
print(f"Final test samples:  {len(X_test)}")


# =========================================================
# INTERNAL TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.10,
    random_state=RANDOM_STATE
)

print(f"Training samples:    {len(X_train)}")
print(f"Validation samples:  {len(X_val)}")


# =========================================================
# SCALE FEATURES
# =========================================================

feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)


# =========================================================
# SCALE TARGET
# =========================================================

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
)

y_val_scaled = target_scaler.transform(
    y_val.values.reshape(-1, 1)
)


# =========================================================
# CONVERT TO PYTORCH TENSORS
# =========================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val_scaled,
    dtype=torch.float32
)


# =========================================================
# MODEL
# =========================================================

model = HousePriceModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# TRAINING WITH EARLY STOPPING
# =========================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0
best_model_state = None

epochs_without_improvement = 0


print("\n=======================================================")
print("TRAINING")
print("=======================================================")

for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    optimizer.zero_grad()

    predictions = model(X_train_tensor)

    train_loss = criterion(
        predictions,
        y_train_tensor
    )

    train_loss.backward()

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

        val_predictions = model(X_val_tensor)

        val_loss = criterion(
            val_predictions,
            y_val_tensor
        )


    train_loss_value = train_loss.item()
    val_loss_value = val_loss.item()

    train_losses.append(train_loss_value)
    val_losses.append(val_loss_value)


    # -------------------------
    # Best model
    # -------------------------

    if val_loss_value < best_val_loss:

        best_val_loss = val_loss_value
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # -------------------------
    # Print progress
    # -------------------------

    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss_value:.4f} | "
            f"Val Loss: {val_loss_value:.4f}"
        )


    # -------------------------
    # Early stopping
    # -------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )

        break


# =========================================================
# RESTORE BEST MODEL
# =========================================================

model.load_state_dict(best_model_state)

print("\n=======================================================")
print("EARLY STOPPING RESULT")
print("=======================================================")

print(f"Best Epoch:          {best_epoch}")
print(f"Best Validation MSE: {best_val_loss:.4f}")


# =========================================================
# VALIDATION METRICS
# =========================================================

model.eval()

with torch.no_grad():

    val_predictions_scaled = model(
        X_val_tensor
    ).numpy()


val_predictions = target_scaler.inverse_transform(
    val_predictions_scaled
).flatten()

val_actual = y_val.values


val_mae = mean_absolute_error(
    val_actual,
    val_predictions
)

val_mse = mean_squared_error(
    val_actual,
    val_predictions
)

val_rmse = np.sqrt(val_mse)

val_r2 = r2_score(
    val_actual,
    val_predictions
)


print("\n=======================================================")
print("VALIDATION PERFORMANCE")
print("=======================================================")

print(f"Validation MAE:  ${val_mae:,.2f}")
print(f"Validation MSE:  {val_mse:,.2f}")
print(f"Validation RMSE: ${val_rmse:,.2f}")
print(f"Validation R²:   {val_r2:.4f}")


# =========================================================
# SAVE LOSS GRAPH
# =========================================================

import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/training_validation_loss.png"
)

plt.close()


# =========================================================
# SAVE MODEL + SCALERS
# =========================================================

torch.save(
    model.state_dict(),
    "models/house_price_model.pth"
)

import pickle

with open(
    "models/feature_scaler.pkl",
    "wb"
) as f:

    pickle.dump(feature_scaler, f)


with open(
    "models/target_scaler.pkl",
    "wb"
) as f:

    pickle.dump(target_scaler, f)


print("\n=======================================================")
print("MODEL SAVED")
print("=======================================================")

print("Model:          models/house_price_model.pth")
print("Feature scaler: models/feature_scaler.pkl")
print("Target scaler:  models/target_scaler.pkl")

print("\nTraining completed successfully.")