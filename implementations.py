import numpy as np
from proj1_helpers import batch_iter
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
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
    
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

        print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
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
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
    
    return (sgd_ws[-1], sgd_losses[-1])
    

# Computing Least Squares
#######################################################

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
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
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return (w, loss)


# Computing Logistic Regression
#######################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    #loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #loss = np.log(1+np.exp(pred))-np.multiply(y,pred) # y.dot(pred)
    #return np.sum(loss)

    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()

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


