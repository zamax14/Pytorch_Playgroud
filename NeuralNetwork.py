import torch
import torch.nn as nn

#Neural Network
class NeuralNetwork(torch.nn.Module):
    def __init__(self, hidden_layer) -> None:
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
