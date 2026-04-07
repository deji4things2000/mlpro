import torch
from fully_connected import FullyConnected


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedLayer, self).__init__()
        size = torch.sqrt(torch.tensor(6 / (in_features + out_features)))
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features).uniform_(-size, size))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(0))

    def forward(self, x):
        return FullyConnected.apply(x, self.weight, self.bias)

    
