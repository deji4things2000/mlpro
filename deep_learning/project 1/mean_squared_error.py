import torch


class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """
        computes the mean squared error between x1 (inputs) and x2 (targets)

        Arguments
        -------
        ctx: a pytorch context object
        x1: (Tensor of size (T x n) where T is the batch size and n is the number of input features.
        x2: (Tensor) of size (T x n)

        Returns
        ------
        y: (scalar) The mean squared error between x1 and x2， averaged over all T * n elements
        """

        # Save for backward
        ctx.save_for_backward(x1, x2)
        ctx.num_elements = x1.numel()
        
        # Compute MSE: mean of (x1 - x2)^2 over all elements
        diff = x1 - x2
        y = torch.mean(diff ** 2)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagates the error with respect to the input arguments

        Arguments
        --------
        ctx: A PyTorch context object
        dzdy:  a scalar (Tensor), the gradient with respect to y

        Returns
        ------
        dzdx1 (Tensor): of size(T x n), the gradients w.r.t x1
        dzdx2 (Tensor): of size(T x n), the gradients w.r.t x2
        """

        x1, x2 = ctx.saved_tensors
        N = ctx.num_elements
        
        # dJ/dx1 = (2/N) * (x1 - x2) * dzdy
        # dJ/dx2 = -(2/N) * (x1 - x2) * dzdy
        diff = x1 - x2
        grad_factor = 2.0 / N * dzdy
        
        dzdx1 = grad_factor * diff
        dzdx2 = -grad_factor * diff

        return dzdx1, dzdx2
