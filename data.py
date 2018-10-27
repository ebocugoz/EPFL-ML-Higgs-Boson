import numpy as np
import matplotlib.pyplot as plt

def ratio_of_nans(data):
    rows, cols = data.shape
    nans = []
    for i in range(cols):
        col = data[:, i]
        count = 0
        for j in range(rows):
            if col[j] == -999:
                count = count + 1
        nans.append(count)
    return np.asarray(nans) / rows

def histograms(data):
    nans = []
    i = 0
    for column in data.T:
        plt.hist(column, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram of Column " + str(i))
        plt.show()
        i = i + 1
        
def log_transform_data(data):
    log_col = [0, 2, 5, 9, 13, 16, 19, 21, 23, 26, 29]
    logColumns = data[:, log_col]
    indices = np.where(logColumns != -999)
    logColumns[indices] = np.log(1 + logColumns[indices])

    data = np.delete(data, log_col, 1)
    data = np.hstack((data, logColumns))
    return data

def clean_data(data):
    data = np.where(data == -999, np.nan, data)

    # Delete columns with nan values
    data = data[:, ~np.all(np.isnan(data), axis=0)]
    # Delete columns with same values
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]

    col_mean = np.nanmedian(data, axis=0)

    # Find indicies that  need to be replaced
    inds = np.where(np.isnan(data))

    # Place column means in the indices. Align the arrays using take
    data[inds] = np.take(col_mean, inds[1])
    return data

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def process_data(data):
    data = log_transform_data(data)
    data = clean_data(data)
    return standardize(data)

def build_model_data(prediction, data):
    """Form (y,tX) to get regression data in matrix form."""
    y = prediction
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def unique_cols(data):
    new_array = [tuple(col) for col in data.T]
    return np.unique(new_array)

def categorize_data(prediction, data):

    indices_0 = np.nonzero(data[:, 22] == 0)[0]
    data_0 = data[indices_0, :]
    pred_0 = prediction[indices_0]

    indices_1 = np.nonzero(data[:, 22] == 1)[0]
    data_1 = data[indices_1, :]
    pred_1 = prediction[indices_1]

    indices_2 = np.nonzero(data[:, 22] > 1)[0]
    data_2 = data[indices_2, :]
    pred_2 = prediction[indices_2]

    return pred_0, pred_1, pred_2, data_0, data_1, data_2, indices_0, indices_1, indices_2

def decategorize_prediction(rows, y_0, y_1, y_2, indices_0, indices_1, indices_2):
    y = np.zeros((rows, 1), dtype=np.float)

    y[indices_0] = y_0
    y[indices_1] = y_1
    y[indices_2] = y_2

    return y