import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        """
        Computes the generalized logistic function

        Arguments
        ---------
        ctx: A PyTorch context object
        x: (Tensor) of size (T x n), the input features
        l, u, and g: (scalar tensors) representing the generalized logistic function parameters.

        Returns
        -------
        y: (Tensor) of size (T x n), the outputs of the generalized logistic operator

        """

        # Save for backward
        ctx.save_for_backward(x, l, u, g)
        
        # Compute y = l + (u - l) / (1 + exp(-g * x))
        exp_neg_gx = torch.exp(-g * x)
        y = l + (u - l) / (1 + exp_neg_gx)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagate the gradients with respect to the inputs

        Arguments
        ----------
        ctx: a PyTorch context object
        dzdy (Tensor): of size (T x n), the gradients with respect to the outputs y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdl, dzdu, and dzdg: the gradients with respect to the generalized logistic parameters
        """

        x, l, u, g = ctx.saved_tensors
        
        # Compute sigmoid part: s = 1 / (1 + exp(-g*x))
        exp_neg_gx = torch.exp(-g * x)
        s = 1 / (1 + exp_neg_gx)
        
        # dy/dx = (u - l) * g * s * (1 - s)
        # Chain rule: dz/dx = dz/dy * dy/dx
        dy_dx = (u - l) * g * s * (1 - s)
        dzdx = dzdy * dy_dx
        
        # dy/dl = 1 - s
        # dz/dl = sum(dz/dy * dy/dl) over all elements
        dy_dl = 1 - s
        dzdl = torch.sum(dzdy * dy_dl)
        
        # dy/du = s
        dy_du = s
        dzdu = torch.sum(dzdy * dy_du)
        
        # dy/dg = (u - l) * x * s * (1 - s)
        dy_dg = (u - l) * x * s * (1 - s)
        dzdg = torch.sum(dzdy * dy_dg)

        return dzdx, dzdl.reshape(1), dzdu.reshape(1), dzdg.reshape(1)
