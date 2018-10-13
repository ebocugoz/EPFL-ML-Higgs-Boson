import numpy as np
def load_data(path_dataset="train.csv",sub_sample=True, add_outlier=False):
    """Load data and convert it to the metrics system."""
   
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1)
  
    
     
    id_ = data[:,0]
    x = np.delete(data,[0,1],axis=1)
    y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: 0 if b"b" in x else 1})
    
    

    return y,x,id_