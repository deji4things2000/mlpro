from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    # Make sure tensors require gradients
    X.requires_grad_(True)
    W.requires_grad_(True)
    B.requires_grad_(True)
    
    # Forward pass
    y = full_connected(X, W, B)
    
    # Create a scalar objective J (mean of y)
    z = y.mean()
    
    # Compute analytical gradients using backward
    z.backward()
    grad_X_analytical = X.grad.clone()
    grad_W_analytical = W.grad.clone()
    grad_B_analytical = B.grad.clone()
    
    # Clear gradients for numerical computation
    X.grad.zero_()
    W.grad.zero_()
    B.grad.zero_()
    
    # Compute DZDY (gradient of z with respect to y)
    # Since z = mean(y), dz/dy = 1/(T*m) for all elements
    DZDY = torch.ones_like(y) / y.numel()
    
    # Compute numerical gradients
    with torch.no_grad():
        # Numerical gradient for X
        grad_X_numerical = torch.zeros_like(X)
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                X_plus = X.clone()
                X_minus = X.clone()
                X_plus[t, i] += DELTA
                X_minus[t, i] -= DELTA
                
                y_plus = full_connected(X_plus, W, B)
                y_minus = full_connected(X_minus, W, B)
                
                # Use chain rule: dJ/dx = sum over (dz/dy * (dy/dx))
                dy_dx = (y_plus - y_minus) / (2 * DELTA)
                grad_X_numerical[t, i] = torch.sum(DZDY * dy_dx)
        
        # Numerical gradient for W
        grad_W_numerical = torch.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.clone()
                W_minus = W.clone()
                W_plus[i, j] += DELTA
                W_minus[i, j] -= DELTA
                
                y_plus = full_connected(X, W_plus, B)
                y_minus = full_connected(X, W_minus, B)
                
                dy_dw = (y_plus - y_minus) / (2 * DELTA)
                grad_W_numerical[i, j] = torch.sum(DZDY * dy_dw)
        
        # Numerical gradient for B
        grad_B_numerical = torch.zeros_like(B)
        for j in range(B.shape[0]):
            B_plus = B.clone()
            B_minus = B.clone()
            B_plus[j] += DELTA
            B_minus[j] -= DELTA
            
            y_plus = full_connected(X, W, B_plus)
            y_minus = full_connected(X, W, B_minus)
            
            dy_db = (y_plus - y_minus) / (2 * DELTA)
            grad_B_numerical[j] = torch.sum(DZDY * dy_db)
    
    # Compute errors
    err_dzdx = torch.max(torch.abs(grad_X_analytical - grad_X_numerical)).item()
    err_dzdw = torch.max(torch.abs(grad_W_analytical - grad_W_numerical)).item()
    err_dzdb = torch.max(torch.abs(grad_B_analytical - grad_B_numerical)).item()
    
    err = {
        'dzdx': err_dzdx,
        'dzdw': err_dzdw,
        'dzdb': err_dzdb
    }
    
    # Use gradcheck for additional verification
    from torch.autograd import gradcheck
    gradcheck_passed = gradcheck(full_connected, (X, W, B), eps=DELTA, atol=TOL)
    
    # Determine if all tests passed
    is_correct = (err_dzdx < TOL) and (err_dzdw < TOL) and (err_dzdb < TOL) and gradcheck_passed

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
    torch.save([tests_passed, errors], 'fully_connected_test_results.pt')