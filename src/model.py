import torch
import torch.nn as nn


class HousePriceModel(nn.Module):

    def __init__(self, hidden1=32, hidden2=16):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(10, hidden1),

            nn.ReLU(),

            nn.Linear(hidden1, hidden2),

            nn.ReLU(),

            nn.Linear(hidden2, 1)
        )


    def forward(self, x):

        return self.network(x)