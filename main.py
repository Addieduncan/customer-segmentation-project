import numpy as np
import pandas as pd
import os

from datatools import make_clean_data, select_features, remove_quantiles, elbow_method


"""
Read Data
"""
DATA_PATH = './raw_data/CreditCard_data.csv'

if os.path.isfile(DATA_PATH):
    print('DATA_PATH is a valid path')
else:
    raise ValueError('DATA_PATH is not valid path')

SAMPLE_SIZE = 10000
data_raw = pd.read_csv(DATA_PATH, nrows = SAMPLE_SIZE)

"""
Preprocess Data
"""
data_clean, _, _ = make_clean_data(data_raw, verbose=False)
data_kept, _, _ = select_features(data_clean, which='basic')

X = data_kept.values.astype(np.float64) # numpy array ready to be clustered

"""
Elbow Method
"""
k_search = np.linspace(start=5, stop=50, num=10)
elbow_method(X, k_search, method = 'KMeans', plot = True)
elbow_method(X, k_search, method = 'GM', plot = True)