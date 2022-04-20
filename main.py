
"""
Names of data after different pathways:

data_valid - after removing NAN values 
data_kept - an array containing the data stored from the valid data, but reduced to the features we 
            desire to keep 
            
X_reduced-  after dimension reduction
X_clean - after quantile outlier reduction 
X_quant - after quantization (distinct from quantile elimination)

"""


import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from datatools import make_clean_data, select_features, \
    remove_quantiles, elbow_method, data_quantization, run_svd,\
        standardize_pd, plot_individual_feature
    
"""
Set Parameters for Processing of Data 
"""


def run_elbow(data_set, method, Kmin=5, Kmax=50, num_K=10, do_transpose = False, feature_select = None):
    k_search = np.linspace(start=Kmin, stop=Kmax, num=num_K, dtype=int)
    optimal_num = elbow_method(data_set, k_search, method=method, plot=True, do_transpose=do_transpose, feature_select= feature_select)

    return optimal_num


if __name__ == "__main__":
    
    """
    1. Set parameters for this file run
    """

    METHOD = 'KMeans' #options are 'GM' or 'KMeans'

    dataset = 'all'  #which dataset; options are 'basic', 'all', or 'freq'
    no_change = False  #Run clustering on cleaned (NaN-removed data). No touching the outliers. 
     
    #Choose only one of these; run clustering on quantized data, or outlier-removed data.
    do_quantize = True
    remove_outliers = not do_quantize

    #Apply SVD on the data (after quantizing / removing outliers)?
    reduce_dim = False
    
    if remove_outliers:
        rescale = True  #If we remove outliers, default behavior is to rescale the data to [0,1] after cleaning
    else: 
        rescale = False


    do_elbow = input('Do you want to run elbow method (y/n): ') # add the option to skip elbow method

    # the list of features to plot for non-transpose problem
    feature_plot = ['CREDIT_LIMIT', 'BALANCE', 'PURCHASES_FREQUENCY', 'PRC_FULL_PAYMENT', 'CASH_ADVANCE_FREQUENCY',\
                    'MINIMUM_PAYMENTS', 'CASH_ADVANCE', 'PURCHASES',  'PURCHASES_INSTALLMENTS_FREQUENCY']

    do_transpose = False

    """
    2. Read In Data
    """
    DATA_PATH = './raw_data/CreditCard_data.csv'
    if os.path.isfile(DATA_PATH):
        print('DATA_PATH is a valid path')
    else:
        raise ValueError('DATA_PATH is not valid path')

    #SAMPLE_SIZE = 10000
    data_raw = pd.read_csv(DATA_PATH)
    
    """
    3. Remove bad data values (NaN) ato obtain valid data. Slice features chosen to obtain desired, valid data. 
    """
    data_valid, _, _ = make_clean_data(data_raw, verbose=False)
    data_kept, feature_kept, _ = select_features(data_valid, which= dataset)
    X = data_kept.values.astype(np.float64) # numpy array ready to be clustered

    """
    4. Elbow Method on Selected Features; Data Otherwise Not Modified 
    """
    if do_transpose:

        X = X.transpose()
        print("\nThis is the transpose problem of shape: {}x{}\n".format(X.shape[0], X.shape[1]))


    if no_change:
        if do_elbow == 'y':
            if do_transpose:
                optimal_num = run_elbow(X, METHOD, Kmin=2, Kmax=13, num_K=12, do_transpose = do_transpose, feature_select = feature_kept)
            else:
                optimal_num = run_elbow(X,METHOD)
        elif do_elbow == 'n':
            optimal_num = input('Input the cluster number to perform clustering')

            if METHOD == 'KMeans':
                cluster = KMeans(n_clusters=int(optimal_num), random_state=0).fit(X)
                cluster_labels = cluster.labels_
            elif METHOD == 'GM':
                cluster = GaussianMixture(n_components=int(optimal_num), random_state=0).fit(X)
                cluster_labels = cluster.predict(X)
            plot_individual_feature(X, cluster_labels, int(optimal_num), feature_kept, feature_plot)
        else:
            raise ValueError('Invalid Response for "do_elbow", only (y/n) allowed')
        """
        5.
            (i)   Remove Outliers or to Quantize All Data; 
            (ii)  Choose Whether To Dimension Reduce; 
            (iii) Run Elbow Method on Resulting Data 
        """

    elif remove_outliers == True:
        
        p = 1 # percent of upper and lower population to be removed
        data_clean, _ = remove_quantiles(data_kept, p)
        assert np.size(data_clean.isna().sum(axis=1).to_numpy().nonzero()[0]) == 0,  "Data still contains NaN"
        
        #If we are rescaling, do the rescaling
        if rescale:
            data_clean = standardize_pd(data_clean)
        else:
            pass
        
        X_clean = data_clean.values.astype(np.float64)

        if do_transpose:
            X_clean = X_clean.transpose()

    
        #X_clean = X_clean - data_clean.min())/(data_clean.max()-data_clean.min())
        if reduce_dim == True: 
            
            #Run SVD and reduce to components which explain at least 99% variance 
            
            desired_var_per = 99
            X_red = run_svd(X_clean, percent_var = desired_var_per)

            if do_elbow == 'y':
                if do_transpose:
                    optimal_num = run_elbow(X_red, METHOD, Kmin=2, Kmax=13, num_K=12, do_transpose=do_transpose,
                                            feature_select=feature_kept)
                else:
                    optimal_num = run_elbow(X_red, METHOD)

            elif do_elbow == 'n':
                optimal_num = input('Input the cluster number to perform clustering: ')
            else:
                raise ValueError('Invalid Response for "do_elbow", only (y/n) allowed')

            if not do_transpose:
                if METHOD == 'KMeans':
                    cluster = KMeans(n_clusters=int(optimal_num), random_state=0).fit(X_red)
                    cluster_labels = cluster.labels_
                elif METHOD == 'GM':
                    cluster = GaussianMixture(n_components=int(optimal_num), random_state=0).fit(X_red)
                    cluster_labels = cluster.predict(X_red)
                plot_individual_feature(X_red, cluster_labels, int(optimal_num), feature_kept, feature_plot)
            
        else:
            if do_elbow == 'y':
                if do_transpose:
                    optimal_num = run_elbow(X_clean, METHOD, Kmin=2, Kmax=13, num_K=12, do_transpose=do_transpose,
                                            feature_select=feature_kept)
                else:
                    optimal_num = run_elbow(X_clean, METHOD)
            elif do_elbow == 'n':
                optimal_num = input('Input the cluster number to perform clustering: ')
            else:
                raise ValueError('Invalid Response for "do_elbow", only (y/n) allowed')

            if not do_transpose:
                if METHOD == 'KMeans':
                    cluster = KMeans(n_clusters=int(optimal_num), random_state=0).fit(X_clean)
                    cluster_labels = cluster.labels_
                elif METHOD == 'GM':
                    cluster = GaussianMixture(n_components=int(optimal_num), random_state=0).fit(X_clean)
                    cluster_labels = cluster.predict(X_clean)
                plot_individual_feature(X_clean, cluster_labels, int(optimal_num), feature_kept, feature_plot)

    elif do_quantize:
        """
        Quantize Data- Convert each feature into integer based on membership in the population quantile 
        """
        
        data_quant, percent_zero = data_quantization(data_kept)
        print('Features have at least the following percentage of being zero:\n', percent_zero)
        X_clean = data_quant.values.astype(np.float64)

        if do_transpose:
            X_clean = X_clean.transpose()
        
        if reduce_dim == True:
            desired_var_per = 99
            X_red = run_svd(X_clean, percent_var = desired_var_per)
            if do_elbow == 'y':
                if do_transpose:
                    optimal_num = run_elbow(X_red, METHOD, Kmin=2, Kmax=13, num_K=12, do_transpose=do_transpose,
                                            feature_select=feature_kept)
                else:
                    optimal_num = run_elbow(X_red, METHOD)

            elif do_elbow == 'n':
                optimal_num = input('Input the cluster number to perform clustering: ')
            else:
                raise ValueError('Invalid Response for "do_elbow", only (y/n) allowed')

            if not do_transpose:
                if METHOD == 'KMeans':
                    cluster = KMeans(n_clusters=int(optimal_num), random_state=0).fit(X_red)
                    cluster_labels = cluster.labels_
                elif METHOD == 'GM':
                    cluster = GaussianMixture(n_components=int(optimal_num), random_state=0).fit(X_red)
                    cluster_labels = cluster.predict(X_red)
                plot_individual_feature(X_red, cluster_labels, int(optimal_num), feature_kept, feature_plot)
        
        elif reduce_dim == False:

            if do_elbow == 'y':
                if do_transpose:
                    optimal_num = run_elbow(X_clean, METHOD, Kmin=2, Kmax=13, num_K=12, do_transpose=do_transpose,
                                            feature_select=feature_kept)
                else:
                    optimal_num = run_elbow(X_clean, METHOD)

            elif do_elbow == 'n':
                optimal_num = input('Input the cluster number to perform clustering: ')
            else:
                raise ValueError('Invalid Response for "do_elbow", only (y/n) allowed')

            if not do_transpose:
                if METHOD == 'KMeans':
                    cluster = KMeans(n_clusters=int(optimal_num), random_state=0).fit(X_clean)
                    cluster_labels = cluster.labels_
                elif METHOD == 'GM':
                    cluster = GaussianMixture(n_components=int(optimal_num), random_state=0).fit(X_clean)
                    cluster_labels = cluster.predict(X_clean)

                plot_individual_feature(X_clean, cluster_labels, int(optimal_num), feature_kept, feature_plot)
