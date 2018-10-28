import numpy as np
from helpers import *
import datetime

# Computing loss / MSE
######################################################

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

# Computing Gradient Descent
#######################################################

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time= {t:.3f} seconds".format(t=exection_time))
    print("Gradient Descent: RMSE Loss = {t}".format(t=np.sqrt(2 * gradient_losses[-1])))
    
    return (gradient_ws[-1], gradient_losses[-1])

# Computing Stochastic Gradient Descent
#######################################################

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        #print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    
    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Define the parameters of the algorithm.
    batch_size = 1

    # Initialization
    w_initial = np.zeros(tx.shape[1])

    # Start SGD.
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Stochastic Gradient Descent: execution time= {t:.3f} seconds".format(t=exection_time))
    print("Stochastic Gradient Descent: RMSE Loss = {t}".format(t=np.sqrt(2 * sgd_losses[-1])))
    
    return (sgd_ws[-1], sgd_losses[-1])
    

# Computing Least Squares
#######################################################

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    start_time = datetime.datetime.now()
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    end_time = datetime.datetime.now()
    
    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Least Squares: execution time= {t:.3f} seconds".format(t=exection_time))
    print("Least Squares: RMSE Loss = {t}".format(t=np.sqrt(2 * loss)))
    
    return (w, loss)

# Computing Ridge Regression
#######################################################

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_poly(x, degree):
    ones_col = np.ones((len(x), 1))
    poly = x
    m, n = x.shape
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    multi_indices = {}
    cpt = 0
    for i in range (n):
        for j in range(i+1,n):
            multi_indices[cpt] = [i,j]
            cpt = cpt+1
    
    gen_features = np.zeros(shape=(m, len(multi_indices)) )

    for i, c in multi_indices.items():
        gen_features[:, i] = np.multiply(x[:, c[0]],x[:, c[1]])

    poly =  np.c_[poly,gen_features]
    poly =  np.c_[ones_col,poly]

    return poly

def ridge_regression(y, tx, lambda_, test = False):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    # Print result
    if not test:
        print("Ridge Regression: RMSE Loss = {t}".format(t=np.sqrt(2 * loss)))
    
    return (w, loss)

def select_hyperparameter_for_ridge_regression(x, y, degree, ratio, seed, lambdas):
    """ridge regression demo."""
    print("Selecting Hyperparameter By Splitting the Data...")

    # split data
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio, seed)
    # form tx
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    # ridge regression with different lambda
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        # ridge regression
        weight, loss = ridge_regression(y_tr, tx_tr, lambda_, test = True)
        rmse_tr.append(np.sqrt(2 * compute_loss(y_tr, tx_tr, weight)))
        rmse_te.append(np.sqrt(2 * compute_loss(y_te, tx_te, weight)))

        # print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)
    
    ind_min = rmse_te.index(min(rmse_te))
    lambda_ = lambdas[ind_min]
    print("Hyperparameter Selection: Lambda = {t}".format(t=lambda_))
    return lambda_ 
    
# Computing Logistic Regression
#######################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
 
    eps = 1e-323
    pred[pred < eps] = eps   
    oneMinusPred = 1 - pred
    oneMinusPred[oneMinusPred < eps] = eps

    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(oneMinusPred))
    return (-y * np.log(pred) - (1 - y) * np.log(oneMinusPred)).mean()

def compute_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.shape[0]
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_log_loss(y, tx, w)
    grad = compute_log_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def logistic_regression(y, tx, max_iter, gamma):
    y = np.expand_dims(y, axis=1)
    losses = []
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)     
        # log info
        if iter % (max_iter/100) == 0:
            print("Current iteration={i}, loss= {l}".format(i=iter, l=loss), end="\r")
            pred = sigmoid(tx.dot(w)) 
        # converge criterion
        losses.append(loss)
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")

    loss = compute_log_loss(y, tx, w)
    print("Logistic Regression: Loss= {l}".format(l=loss))
    
    y = np.squeeze(y)
    return (w, loss)

# Computing Regularized Logistic Regression
#######################################################

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = compute_log_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_,max_iter, gamma):
    y = np.expand_dims(y, axis=1)
    losses = []
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % (max_iter/100) == 0:
            print("Current iteration={i}, loss= {l}".format(i=iter, l=loss), end="\r")
        # converge criterion
        losses.append(loss)
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent")
    
    loss = compute_log_loss(y, tx, w)
    print("Regularized Logistic Regression: Loss= {l}".format(l=loss))
    
    y = np.squeeze(y)
    return (w, loss)


# Computing Cross Validation
#######################################################

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # get k'th subgroup in test, others in train
    x_test = x[k_indices[k].ravel()]
    x_train = x[k_indices[np.arange(len(k_indices))!=k].ravel()]
                            
    y_test = y[k_indices[k].ravel()]
    y_train = y[k_indices[np.arange(len(k_indices))!=k].ravel()]
    
    # form data with polynomial degree:
    poly_test = build_poly(x_test,degree)
    poly_train = build_poly(x_train,degree)

    # ridge regression:
    weights, loss = ridge_regression(y_train, poly_train, lambda_, test = True)
    
    # calculate the loss for train and test data:
    rmse_tr = (np.sqrt(2 * compute_loss(y_train, poly_train, weights)))
    rmse_te = (np.sqrt(2 * compute_loss(y_test, poly_test, weights)))
    
    return rmse_tr, rmse_te

def select_hyperparameter_with_cross_validation(y, x, seed, degree, k_fold, step, lambdas):
    print("Selecting Hyperparameter By Using {k}-fold Cross Validation...".format(k = k_fold))
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    total_iter = step * k_fold
    curr_iter = 0
    for ind, lambda_ in enumerate(lambdas):
        # ridge regression
        for k in range(4):
            if curr_iter % (total_iter/50) == 0: 
                print("Progress: %" + str(curr_iter * 100 / total_iter), end="\r")
            curr_iter = curr_iter + 1
            [a,b] = cross_validation(y,x,k_indices,k,lambda_,degree)
            rmse_tr.append(a)
            rmse_te.append(b)
    rmse_tr = np.asarray(rmse_tr).reshape(-1,4)
    rmse_te = np.asarray(rmse_te).reshape(-1,4)
    rmse_tr = np.mean(rmse_tr,axis=1)
    rmse_te = np.mean(rmse_te,axis=1)
    
    # cross validation visualization:
    cross_validation_visualization(lambdas,rmse_tr,rmse_te)
    
    ind_min = np.argmin(rmse_te)
    lambda_ = lambdas[ind_min]
    
    print("Hyperparameter Selection: Lambda = {t}".format(t=lambda_))
    return lambda_

