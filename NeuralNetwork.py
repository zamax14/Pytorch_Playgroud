import torch

#Neural Network
class NeuralNetwork(torch.nn.Module):
    def __init__(self, hidden_layer) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_layer),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_layer, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, tensor):
        return self.model(tensor)