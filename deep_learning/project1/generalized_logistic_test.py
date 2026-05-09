from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%

        # Test forward mode against torch.tanh
    # For tanH: l = -1, u = 1, g = 2
    y_custom = generalized_logistic(X, L, U, G)
    y_tanh = torch.tanh(X)
    
    err_y = torch.max(torch.abs(y_custom - y_tanh)).item()
    
    # Make sure tensors require gradients
    X.requires_grad_(True)
    # L, U, G are scalars - need requires_grad for gradcheck
    L.requires_grad_(True)
    U.requires_grad_(True)
    G.requires_grad_(True)
    
    # Forward pass
    y = generalized_logistic(X, L, U, G)
    
    # Create scalar objective (mean of y)
    z = y.mean()
    
    # Compute analytical gradients
    z.backward()
    grad_X_analytical = X.grad.clone()
    grad_L_analytical = L.grad.clone()
    grad_U_analytical = U.grad.clone()
    grad_G_analytical = G.grad.clone()
    
    # Clear gradients
    X.grad.zero_()
    L.grad.zero_()
    U.grad.zero_()
    G.grad.zero_()
    
    # DZDY is 1/(T*n) since z = mean(y)
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
                
                y_plus = generalized_logistic(X_plus, L, U, G)
                y_minus = generalized_logistic(X_minus, L, U, G)
                
                dy_dx = (y_plus - y_minus) / (2 * DELTA)
                grad_X_numerical[t, i] = torch.sum(DZDY * dy_dx)
        
        # Numerical gradient for L
        L_plus = L.clone() + DELTA
        L_minus = L.clone() - DELTA
        y_plus = generalized_logistic(X, L_plus, U, G)
        y_minus = generalized_logistic(X, L_minus, U, G)
        dy_dl = (y_plus - y_minus) / (2 * DELTA)
        grad_L_numerical = torch.sum(DZDY * dy_dl)
        
        # Numerical gradient for U
        U_plus = U.clone() + DELTA
        U_minus = U.clone() - DELTA
        y_plus = generalized_logistic(X, L, U_plus, G)
        y_minus = generalized_logistic(X, L, U_minus, G)
        dy_du = (y_plus - y_minus) / (2 * DELTA)
        grad_U_numerical = torch.sum(DZDY * dy_du)
        
        # Numerical gradient for G
        G_plus = G.clone() + DELTA
        G_minus = G.clone() - DELTA
        y_plus = generalized_logistic(X, L, U, G_plus)
        y_minus = generalized_logistic(X, L, U, G_minus)
        dy_dg = (y_plus - y_minus) / (2 * DELTA)
        grad_G_numerical = torch.sum(DZDY * dy_dg)
    
    # Compute errors
    err_dzdx = torch.max(torch.abs(grad_X_analytical - grad_X_numerical)).item()
    err_dzdl = torch.abs(grad_L_analytical - grad_L_numerical).item()
    err_dzdu = torch.abs(grad_U_analytical - grad_U_numerical).item()
    err_dzdg = torch.abs(grad_G_analytical - grad_G_numerical).item()
    
    err = {
        'y': err_y,
        'dzdx': err_dzdx,
        'dzdl': err_dzdl,
        'dzdu': err_dzdu,
        'dzdg': err_dzdg
    }
    
    # Use gradcheck for additional verification
    from torch.autograd import gradcheck
    gradcheck_passed = gradcheck(generalized_logistic, (X, L, U, G), eps=DELTA, atol=TOL2)
    
    # Determine if all tests passed
    is_correct = (err_y < TOL1) and (err_dzdx < TOL2) and (err_dzdl < TOL2) and \
                 (err_dzdu < TOL2) and (err_dzdg < TOL2) and gradcheck_passed



    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)
    torch.save([test_passed, errors], 'generalized_logistic_test_results.pt')