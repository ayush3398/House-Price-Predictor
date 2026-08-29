import pandas as pd
import matplotlib.pyplot as plt


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


# Calculate correlations
correlations = df[features + [target]].corr()[target]

# Remove SalePrice itself
correlations = correlations.drop(target)

# Sort by absolute correlation
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)


print("\nFeature Relationship with SalePrice")
print("------------------------------------")

for feature, correlation in correlations.items():
    print(f"{feature:15s}: {correlation:.3f}")


# Plot correlations
plt.figure(figsize=(10, 6))

correlations.sort_values().plot(
    kind="barh"
)

plt.xlabel("Correlation with SalePrice")
plt.ylabel("Feature")
plt.title("Feature Relationship with House Price")

plt.tight_layout()

plt.savefig(
    "results/feature_correlations.png"
)

plt.show()