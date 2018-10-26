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

def uniqueCols(data):
    new_array = [tuple(col) for col in data.T]
    return np.unique(new_array)

def categorizeData(prediction, data):

    indices_0 = np.nonzero(data[:, 22] == 0)[0]
    data_0 = data[indices_0, :]
    pred_0 = prediction[indices_0]

    indices_1 = np.nonzero(data[:, 22] == 1)[0]
    data_1 = data[indices_1, :]
    pred_1 = prediction[indices_1]

    indices_2 = np.nonzero(data[:, 22] == 2)[0]
    data_2 = data[indices_2, :]
    pred_2 = prediction[indices_2]

    indices_3 = np.nonzero(data[:, 22] == 3)[0]
    data_3 = data[indices_3, :]
    pred_3 = prediction[indices_3]

    return pred_0, pred_1, pred_2, pred_3, data_0, data_1, data_2, data_3, indices_0, indices_1, indices_2, indices_3

def decategorizePrediction(rows, y_0, y_1, y_2, y_3, indices_0, indices_1, indices_2, indices_3):
    y = np.zeros((rows, 1), dtype=np.float)

    y[indices_0] = y_0
    y[indices_1] = y_1
    y[indices_2] = y_2
    y[indices_3] = y_3

    return y