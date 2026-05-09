import torch


class FullyConnected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        """
        Computes the output of the fully_connected function given in the assignment

        Arguments
        ---------
        ctx: a PyTorch context object
        x (Tensor): of size (T x n), the input features
        w (Tensor): of size (n x m), the weights
        b (Tensor): of size (m), the biases

        Returns
        -----
        y (Tensor): of size (T x m), the outputs of the fully_connected operator
        """

        ctx.save_for_backward(x, w, b)
        
        # Compute y = x @ w + b
        # x: (T x n), w: (n x m) -> x @ w: (T x m)
        # b: (m) broadcasts to (T x m)
        y = torch.mm(x, w) + b

        return y

    @staticmethod
    def backward(ctx, dz_dy):
        """
        back-propagates the gradients with respect to the inputs
        ctx: a PyTorch context object.
        dz_dy (Tensor): of size (T x m), the gradients with respect to the output argument y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdw (Tensor): of size (n x m), the gradients with respect to w
        dzdb (Tensor): of size (m), the gradients with respect to b
        """

        # Retrieve saved tensors
        x, w, b = ctx.saved_tensors
        
        # dz/dx = dz/dy @ w^T
        # dz/dy: (T x m), w^T: (m x n) -> dzdx: (T x n)
        dzdx = torch.mm(dz_dy, w.t())
        
        # dz/dw = x^T @ dz/dy
        # x^T: (n x T), dz/dy: (T x m) -> dzdw: (n x m)
        dzdw = torch.mm(x.t(), dz_dy)
        
        # dz/db = sum over batch dimension of dz/dy
        # dz/dy: (T x m) -> sum dim 0: (m)
        dzdb = torch.sum(dz_dy, dim=0)

        return dzdx, dzdw, dzdb
