# House Price Predictor

A machine learning project that predicts house sale prices using selected features from the Ames Housing Dataset.

## Project Objective

The objective of this project is to build a neural network regression model that can learn relationships between house characteristics and their sale prices.

The system can:

- Train a neural network on housing data
- Evaluate model performance
- Analyze prediction errors
- Predict the price of a new house from user-provided features

## Dataset

The project uses the Ames Housing Dataset.

- Total samples: 2930
- Total columns: 82
- Target variable: `SalePrice`

## Selected Features

The model uses 10 numerical features:

1. Overall Qual
2. Gr Liv Area
3. Garage Cars
4. Total Bsmt SF
5. 1st Flr SF
6. Full Bath
7. TotRms AbvGrd
8. Year Built
9. Garage Area
10. Lot Area

Missing numerical values are handled using median imputation.

## Model Architecture

A feed-forward neural network is used for regression.

```text
Input Layer: 10 features
        ↓
Dense Layer: 32 neurons
        ↓
ReLU
        ↓
Dropout: 0.2
        ↓
Dense Layer: 16 neurons
        ↓
ReLU
        ↓
Dropout: 0.2
        ↓
Output Layer: 1 neuron
