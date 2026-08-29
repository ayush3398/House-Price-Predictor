import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# Create X and y
X = df[features].copy()
y = df[target].copy()


# Fill missing values
X = X.fillna(X.median())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Normalize the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Print results
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nNumber of features:", X_train.shape[1])

print("\nFirst training sample:")
print(X_train[0])

print("\nFirst training target:")
print(y_train.iloc[0])