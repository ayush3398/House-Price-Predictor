import torch
import torch.nn as nn


class HousePriceModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Input layer → first hidden layer
            nn.Linear(10, 32),

            # Activation
            nn.ReLU(),

            # Dropout to reduce overfitting
            nn.Dropout(0.2),

            # Second hidden layer
            nn.Linear(32, 16),

            # Activation
            nn.ReLU(),

            # Another dropout layer
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)