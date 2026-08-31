# House Price Prediction Using Neural Networks

## 1. Project Overview

This project predicts house sale prices using a feed-forward neural network.

The project uses the Ames Housing dataset, which contains information about residential properties such as house quality, living area, garage size, basement area, number of bathrooms, and lot size.

The goal is to build a simple and understandable Artificial Intelligence / Machine Learning model that can learn the relationship between property characteristics and house prices.

---

## 2. Dataset

The project uses the Ames Housing dataset.

The dataset contains 2,930 houses and 82 columns.

The target variable is:

- SalePrice

Ten property features were selected for the neural network:

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

These features were selected because they represent important characteristics of a property and have meaningful relationships with house prices.

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

### Missing values

Missing numerical values in the selected features were replaced using the median of each feature.

### Train/Test Split

The dataset was divided into training and testing data.

A validation set was also created from the training data to monitor the model during training.

### Feature Normalization

StandardScaler was used to normalize the input features.

This was important because the features have very different numerical ranges.

### Target Normalization

SalePrice was also normalized during training.

After prediction, the values were converted back to the original dollar scale using the inverse transformation.

---

## 4. Neural Network Architecture

The project uses a simple feed-forward neural network.

Architecture:

10 input features
        ↓
32 neurons
        ↓
ReLU
        ↓
16 neurons
        ↓
ReLU
        ↓
1 output neuron

The output represents the predicted house price.

The model was intentionally kept small and understandable because the goal was to build a student-level AI/ML project rather than a highly complex system.

---

## 5. Training

The model was trained using:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 200
- Gradient clipping: maximum norm of 1.0
- Activation function: ReLU

During training, both training loss and validation loss were recorded.

---

## 6. Evaluation

The model was evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual prices.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater importance to larger errors.

### R² Score

Measures how much of the variation in house prices is explained by the model.

### Final Results

| Metric | Result |
|---|---:|
| MAE | $20,173.83 |
| RMSE | $34,369.08 |
| R² | 0.8527 |

The model explains approximately 85.27% of the variation in house prices on the test set.

---

## 7. Visualizations

The project includes:

### Actual vs Predicted Prices

This plot compares the actual house prices with the prices predicted by the neural network.

Points closer to the diagonal line represent more accurate predictions.

### Residual Plot

Residuals are calculated as:

Actual Price - Predicted Price

The residual plot helps identify patterns and unusually large prediction errors.

### Training and Validation Loss

This graph shows how the training and validation loss changed over the training epochs.

### Feature Correlation

Correlation analysis was used to understand the relationship between the selected property features and SalePrice.

---

## 8. Model Interpretation

The strongest relationship with SalePrice was observed for:

| Feature | Correlation |
|---|---:|
| Overall Qual | 0.799 |
| Gr Liv Area | 0.707 |
| Garage Cars | 0.648 |
| Garage Area | 0.640 |
| Total Bsmt SF | 0.632 |

Overall Quality had the strongest relationship with house price.

This suggests that property quality and usable living space are particularly important characteristics associated with house prices.

Correlation does not imply causation and does not directly represent the neural network's internal feature importance.

---

## 9. Project Structure

```text
House_predictor/
│
├── data/
│   └── AmesHousing.csv
│
├── models/
│   ├── house_price_model.pth
│   └── target_scaler.pkl
│
├── results/
│   ├── training_validation_loss.png
│   ├── residual_plot.png
│   └── feature_correlations.png
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── residuals.py
│   ├── interpretation.py
│   └── test_data.py
│
├── requirements.txt
└── README.md