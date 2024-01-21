import pandas as pd
import numpy as np


def balance_data(X, y, undersampling=True, seed=111):
    """
    Function to balance data
    """
    n_0 = y.value_counts()[0]
    n_1 = y.value_counts()[1]

    n_lower = y.value_counts().min()
    n_upper = y.value_counts().max()
    
    np.random.seed(seed)
    
    if undersampling:
        # undersampling    
        index_0 = np.random.choice(y[y==0].index, n_lower, replace=False)
        index_1 = np.random.choice(y[y==1].index, n_lower, replace=False)
    
        y_u = y.iloc[ index_0.tolist()+index_1.tolist() ].sort_index()
        
        X_u = X.iloc[ index_0.tolist()+index_1.tolist() ].sort_index()
    else:
        # oversampling
        if n_0 < n_1:
            index_0 = np.random.choice(y[y==0].index, n_1, replace=True)
            index_1 = np.random.choice(y[y==1].index, n_1, replace=False)
        else:
            index_0 = np.random.choice(y[y==0].index, n_0, replace=False)
            index_1 = np.random.choice(y[y==1].index, n_0, replace=True)
    
        y_u = y.iloc[ index_0.tolist()+index_1.tolist() ].sort_index()
        
        X_u = X.iloc[ index_0.tolist()+index_1.tolist() ].sort_index()

    return X_u, y_u