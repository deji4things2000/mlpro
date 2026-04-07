import torch
from generalized_logistic import GeneralizedLogistic


class GeneralizedLogisticLayer(torch.nn.Module):
    def __init__(self, non_linearity='tanH'):
        super(GeneralizedLogisticLayer, self).__init__()

        self._non_linearity = non_linearity

    def forward(self, x):
        if self._non_linearity == "tanH":
            l = torch.tensor([-1], dtype=x.dtype, requires_grad=True)
            g = torch.tensor([2], dtype=x.dtype, requires_grad=True)
        elif self._non_linearity == 'sigmoid':
            l = torch.tensor([0], dtype=x.dtype, requires_grad=True)
            g = torch.tensor([1], dtype=x.dtype, requires_grad=True)
        else:
            raise ValueError("Error: unknown generalized logistic non-linearity")

        u = torch.tensor([1], dtype=x.dtype, requires_grad=True)
        return GeneralizedLogistic.apply(x, l, u, g)

    
