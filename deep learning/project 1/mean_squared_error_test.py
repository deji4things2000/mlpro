from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 48 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%

        # Make sure tensors require gradients
    X1.requires_grad_(True)
    X2.requires_grad_(True)
    
    # Forward pass
    y = mean_squared_error(X1, X2)
    
    # Compute analytical gradients using backward
    y.backward()
    grad_X1_analytical = X1.grad.clone()
    grad_X2_analytical = X2.grad.clone()
    
    # Clear gradients
    X1.grad.zero_()
    X2.grad.zero_()
    
    # DZDY is 1 since y is a scalar and we're backpropagating from it directly
    DZDY = torch.tensor(1.0)
    
    # Compute numerical gradients
    with torch.no_grad():
        # Numerical gradient for X1
        grad_X1_numerical = torch.zeros_like(X1)
        for t in range(X1.shape[0]):
            for i in range(X1.shape[1]):
                X1_plus = X1.clone()
                X1_minus = X1.clone()
                X1_plus[t, i] += DELTA
                X1_minus[t, i] -= DELTA
                
                y_plus = mean_squared_error(X1_plus, X2)
                y_minus = mean_squared_error(X1_minus, X2)
                
                grad_X1_numerical[t, i] = (y_plus - y_minus) / (2 * DELTA)
        
        # Numerical gradient for X2
        grad_X2_numerical = torch.zeros_like(X2)
        for t in range(X2.shape[0]):
            for i in range(X2.shape[1]):
                X2_plus = X2.clone()
                X2_minus = X2.clone()
                X2_plus[t, i] += DELTA
                X2_minus[t, i] -= DELTA
                
                y_plus = mean_squared_error(X1, X2_plus)
                y_minus = mean_squared_error(X1, X2_minus)
                
                grad_X2_numerical[t, i] = (y_plus - y_minus) / (2 * DELTA)
    
    # Compute errors
    err_dzdx1 = torch.max(torch.abs(grad_X1_analytical - grad_X1_numerical)).item()
    err_dzdx2 = torch.max(torch.abs(grad_X2_analytical - grad_X2_numerical)).item()
    
    err = {
        'dzdx1': err_dzdx1,
        'dzdx2': err_dzdx2
    }
    
    # Use gradcheck for additional verification
    from torch.autograd import gradcheck
    gradcheck_passed = gradcheck(mean_squared_error, (X1, X2), eps=DELTA, atol=TOL)
    
    # Determine if all tests passed
    is_correct = (err_dzdx1 < TOL) and (err_dzdx2 < TOL) and gradcheck_passed



    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
    torch.save([tests_passed, errors], 'mean_squared_error_test_results.pt')