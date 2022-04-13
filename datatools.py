import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_clean_data(pd_data,verbose = False):
    """
    -Given any dataframe, removes rows with nan or none, examining 
    feature by feature 
    -This probably can be done by doing the whole dataframe simultaneously 
    """
    #infer the features we have kept by checking columns of the input dataframe
    
    #Needed: check all features for missing values and remove everybody with missing. 
    #check remaining percentage 
    
    features = pd_data.columns 
    missing_dict = dict()
    # bad_rows = list()

    bad_rows_index = pd_data.isna().sum(axis=1).to_numpy().nonzero()
    bad_feature_index = pd_data.isna().sum().to_numpy().nonzero()
    bad_feature = pd_data.columns[bad_feature_index]

    for ft in bad_feature:
        #Calculate the percentage of that feature which was True under .isnull()
        num_missing_bool = pd_data.isna().sum()[ft]
        missing_dict[ft] =  num_missing_bool  / len(pd_data[ft])

        if verbose:
            print("Issue Feature:\n", ft, '\n', '\n Num of null=', num_missing_bool, '\n\n')
        else:
            pass

    # for ft in features:
    #     pd_data.isna().sum()
    #     feature_series = pd_data[ft]
    #     missing_bool = feature_series.isnull()
    #     bad_indices = feature_series.index[missing_bool]
    #     #Calculate the percentage of that feature which was True under .isnull()
    #     missing_dict[ft] = 100*float(np.sum(missing_bool)/feature_series.shape[0])
    #
    #
    #     if not bad_indices.empty:
    #         if verbose:
    #             print("Issue Feature:\n", ft,'\n', bad_indices, '\n Num of null=', len(bad_indices), '\n\n')
    #             bad_rows += list(bad_indices)
    #             print('Here are Nan Indices:', bad_indices)
    #         else:
    #             pass
            
    #Total percentage(s) of data removed
    if verbose:
        # print('Here are Nan Row Indices:', bad_rows_index[0], '\n') #maybe we don't need but I added here
        print('Total Number of Removed Row Instances = ', len(bad_rows_index[0]),'\n ')
        print('Percentage of Removed Features: \n',missing_dict)
    #Eliminate duplicates and sort 
    # bad_rows = list(set(bad_rows))
    # bad_rows.sort()

    # Get rid of rows containing null or empty
    # clean_data = pd_data.drop(bad_rows)
    clean_data = pd_data.drop(bad_rows_index[0])

    # Check if clean
    assert np.size(clean_data.isna().sum(axis=1).to_numpy().nonzero()[0]) == 0, "Clean data still contains NaN"

    #Check the number of resulting data points 
    # if verbose:
    #     print('Here is shape of original data:',data.shape,'\n\n')
    #     print('Here is shape of the clean data:', data_clean.shape,\
    #           '\n Number of Removed Instances =',len(bad_rows))
        
    return clean_data, missing_dict, bad_rows_index

def select_features(pd_data, which = 'basic'):
    """
    pd_data: should be the raw, completely unprocessed feature data 
    
    which:: <string> - determine which set of features to use. Option: 'basic', 'freq', 'all'
    """
    features = pd_data.columns 
    ft_keep = list()
    
    #Here are the features related to the absolute amount of money; some of them can be used as LABEL
    ft_basic =['BALANCE', 'PURCHASES',
           'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
           'CREDIT_LIMIT', 'PAYMENTS',
           'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
    #the last one is the percent of full payment being paid

    #Here are the features related to the frequency
    ft_freq= ['BALANCE_FREQUENCY','PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
           'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']

    #Define which features to keep 
    if which =='basic':
        ft_keep = ft_basic[:]
        print('Here are the selected features (related to dollars):\n', ft_basic, '\n')
    if which == 'freq':
        ft_keep = ft_freq[:]
        print('Here are the selected features (related to frequency): \n', ft_freq, '\n')
    elif which == 'all':
        ft_keep = ft_basic[:] + ft_freq[:]
        print('Here are the selected features (all of them): \n', ft_keep, '\n')
    print('Features retained are: '+which+'\n')
    
    #Here are the features haven't been used
    ft_unused = set(list(features))-set(ft_keep)
    ft_unused = list(ft_unused)

    # print('Here are the features related to dollars:\n', ft_basic,'\n')
    # print('Here are the features related to frequency: \n',ft_freq,'\n')
    print('Here are the features not used: \n', ft_unused)
    
    #Now slice the data according to the desired features
    keep_data = pd_data[ft_keep]

    return keep_data, ft_keep[:], list(ft_unused)

def remove_quantiles(pd_data,p = 1):
    percentile = p
    quantile = percentile / 100 
    remove_indices = list()
    # feature_stats = dict()
    
    for feature in pd_data.columns:
        feature_series = pd_data[feature]
        quantile_filter = np.quantile(feature_series,[quantile,1-quantile])
        feature_outside = feature_series[(feature_series < quantile_filter[0]) | (feature_series > quantile_filter[1])]
        # outside_indices = feature_outside.index
        remove_indices += list(feature_outside.index)
    remove_indices = list(set(remove_indices))
    remove_indices.sort()
    
    pd_data_reduced = pd_data.drop(remove_indices)
    
    #Calculate what percent of total data is captured in these indices
    percent_removed = 100*(len(remove_indices)/pd_data.index.shape[0])
    print('Percent of Data Removed Across These Quantiles Is: ', percent_removed)
    
    return pd_data_reduced, percent_removed

def elbow_method(X, k_search, method = 'KMeans', plot = True):
    """
    Elbow Method for different clustering methods with all metrics shown
    :param X: (np,ndarray NxD) data matrix 
    :param k_search: (np.ndarray) list containing the number of clusters to compare over
    :param method: (string) "Kmeans" or "GM" specify the clustering techniques
    :param plot: (boolean) if plotting the results or not
    :return: 
    """
    silh_score = np.zeros(len(k_search))
    CHindex_score = silh_score.copy()
    DBindex_score = silh_score.copy()
    if method == 'KMeans' or method == 'GM':
        pass
    else:
        raise ValueError('method is not a valid method (only "kmeans" or "gm" is available)')
    # if method == "kmeans":
    #     SoS = silh_score.copy()
    SoS = silh_score.copy()
    print("Running Elbow Method...")
    for (i, k) in tqdm(enumerate(k_search), total=len(k_search)):
        if method == 'KMeans':
            kmeans = KMeans(n_clusters=int(k), random_state=0).fit(X)
            kmeans_label = kmeans.labels_
            SoS[i] = kmeans.inertia_
            silh_score[i] = metrics.silhouette_score(X, kmeans_label, metric='euclidean')
            CHindex_score[i] = metrics.calinski_harabasz_score(X, kmeans_label)
            DBindex_score[i] = metrics.davies_bouldin_score(X, kmeans_label)
        elif method == 'GM':
            gm = GaussianMixture(n_components=int(k), random_state=0).fit(X)
            gm_label = gm.predict(X)
            silh_score[i] = metrics.silhouette_score(X, gm_label, metric='euclidean')
            CHindex_score[i] = metrics.calinski_harabasz_score(X, gm_label)
            DBindex_score[i] = metrics.davies_bouldin_score(X, gm_label)

    metric_list = [silh_score/np.max(np.abs(silh_score)), CHindex_score/np.max(np.abs(CHindex_score)),
                   DBindex_score/np.max(np.abs(DBindex_score)), SoS/(max(np.max(SoS),0.1))]
    metric_legend = ['Silhouette', 'CHindex', 'DBindex', 'SoS']

    if plot:
        if method == 'KMeans':
            m = 4
        elif method == 'GM':
            m = 3
        # plt.figure()
        # plt.figure(figsize=(6, 6))
        Markers = ['+', 'o', '*', 'x']
        for i in range(m):
            plt.plot(k_search, metric_list[i], marker = Markers[i])
        plt.xlabel(r'Number of clusters $k$', fontsize=20, fontname="Times New Roman", fontweight='bold')
        plt.ylabel('Metric Score (Normalized)', fontsize=20, fontname="Times New Roman", fontweight='bold')
        plt.title('Evaluation of {} clustering'.format(method), fontsize=22, fontweight='bold')
        plt.legend(metric_legend, loc='best')
        plt.show()
    return





if __name__ == "__main__":
    
    pass 