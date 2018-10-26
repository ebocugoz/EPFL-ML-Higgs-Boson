import numpy as np

def clean_data(data):
    data = np.where(data==-999, np.nan, data)

    col_mean = np.nanmedian(data, axis=0)
    
    #col_stddev = np.sqrt(np.nanvar(data, axis=0))
    #stat = np.vstack((col_mean, col_stddev))

    #Find indicies that you need to replace
    inds = np.where(np.isnan(data))

    #Place column means in the indices. Align the arrays using take
    data[inds] = np.take(col_mean, inds[1])
    return data