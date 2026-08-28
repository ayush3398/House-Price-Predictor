import pandas as pd

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

# Fill missing values with median
X = X.fillna(X.median())

# Check missing values after cleaning
print("Missing values after cleaning:")
print(X.isnull().sum())

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())