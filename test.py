import numpy as np
from implementations import *
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    x_test = x[k_indices[k].ravel()]
    x_train = x[k_indices[np.arange(len(k_indices))!=k].ravel()]
    
                            
    y_test = y[k_indices[k].ravel()]
    y_train = y[k_indices[np.arange(len(k_indices))!=k].ravel()]
 
    
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    poly_test = build_poly(x_test,degree)
    poly_train = build_poly(x_train,degree)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    weight = ridge_regression(y_train, poly_train, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    
    rmse_tr = (np.sqrt(2 * compute_loss(y_train, poly_train, weight)))
    rmse_te = (np.sqrt(2 * compute_loss(y_test, poly_test, weight)))
    
    
    return rmse_tr, rmse_te