import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from kneed import KneeLocator
from matplotlib.colors import ListedColormap



def standardize_pd(Xin):
    """
    Parameters
    ----------
    Xin : <pd.DataFrame> dataframe containing the dataset 

    Returns
    -------
    <pd.DataFrame> the column-wise standardized dataset
    """
    
    return Xin-Xin.min()/(Xin.max()-Xin.min())
    

def make_clean_data(pd_data, verbose=False):
    """
    -Given any dataframe, removes rows with nan or none, examining 
    feature by feature 
    -This probably can be done by doing the whole dataframe simultaneously 
    """
    # infer the features we have kept by checking columns of the input dataframe

    # Needed: check all features for missing values and remove everybody with missing.
    # check remaining percentage

    features = pd_data.columns
    missing_dict = dict()
    # bad_rows = list()

    bad_rows_index = pd_data.isna().sum(axis=1).to_numpy().nonzero()
    bad_feature_index = pd_data.isna().sum().to_numpy().nonzero()
    bad_feature = pd_data.columns[bad_feature_index]

    for ft in bad_feature:
        # Calculate the percentage of that feature which was True under .isnull()
        num_missing_bool = pd_data.isna().sum()[ft]
        missing_dict[ft] = num_missing_bool / len(pd_data[ft])

        if verbose:
            print("Issue Feature:\n", ft, '\n',
                  '\n Num of null=', num_missing_bool, '\n\n')
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

    # Total percentage(s) of data removed
    if verbose:
        # print('Here are Nan Row Indices:', bad_rows_index[0], '\n') #maybe we don't need but I added here
        print('Total Number of Removed Row Instances = ',
              len(bad_rows_index[0]), '\n ')
        print('Percentage of Removed Features: \n', missing_dict)
    # Eliminate duplicates and sort
    # bad_rows = list(set(bad_rows))
    # bad_rows.sort()

    # Get rid of rows containing null or empty
    # clean_data = pd_data.drop(bad_rows)
    clean_data = pd_data.drop(bad_rows_index[0])

    # Check if clean
    assert np.size(clean_data.isna().sum(axis=1).to_numpy().nonzero()[
                   0]) == 0, "Clean data still contains NaN"

    # Check the number of resulting data points
    # if verbose:
    #     print('Here is shape of original data:',data.shape,'\n\n')
    #     print('Here is shape of the clean data:', data_clean.shape,\
    #           '\n Number of Removed Instances =',len(bad_rows))

    return clean_data, missing_dict, bad_rows_index


def select_features(pd_data, which='basic'):
    """
    pd_data: should be the raw, completely unprocessed feature data 

    which:: <string> - determine which set of features to use. Option: 'basic', 'freq', 'all'
    """
    features = pd_data.columns
    ft_keep = list()

    # Here are the features related to the absolute amount of money; some of them can be used as LABEL
    ft_basic = ['BALANCE', 'PURCHASES',
                'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                'CREDIT_LIMIT', 'PAYMENTS',
                'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
    # the last one is the percent of full payment being paid

    # Here are the features related to the frequency
    ft_freq = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
               'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']

    # Define which features to keep
    if which == 'basic':
        ft_keep = ft_basic[:]
        print('Here are the selected features (related to dollars):\n', ft_basic, '\n')
    if which == 'freq':
        ft_keep = ft_freq[:]
        print('Here are the selected features (related to frequency): \n', ft_freq, '\n')
    elif which == 'all':
        ft_keep = ft_basic[:] + ft_freq[:]
        print('Here are the selected features (all of them): \n', ft_keep, '\n')
    print('Features retained are: '+which+'\n')

    # Here are the features haven't been used
    ft_unused = set(list(features))-set(ft_keep)
    ft_unused = list(ft_unused)

    # print('Here are the features related to dollars:\n', ft_basic,'\n')
    # print('Here are the features related to frequency: \n',ft_freq,'\n')
    print('Here are the features not used: \n', ft_unused)

    # Now slice the data according to the desired features
    keep_data = pd_data[ft_keep]

    return keep_data, ft_keep[:], list(ft_unused)


def remove_quantiles(pd_data, p=1):
    percentile = p
    quantile = percentile / 100
    remove_indices = list()
    # feature_stats = dict()

    for feature in pd_data.columns:
        feature_series = pd_data[feature]
        quantile_filter = np.quantile(feature_series, [quantile, 1-quantile])
        feature_outside = feature_series[(feature_series < quantile_filter[0]) | (
            feature_series > quantile_filter[1])]
        # outside_indices = feature_outside.index
        remove_indices += list(feature_outside.index)
    remove_indices = list(set(remove_indices))
    remove_indices.sort()

    pd_data_reduced = pd_data.drop(remove_indices)

    # Calculate what percent of total data is captured in these indices
    percent_removed = 100*(len(remove_indices)/pd_data.index.shape[0])
    print('Percent of Data Removed Across These Quantiles Is: ', percent_removed)

    return pd_data_reduced, percent_removed


def run_svd(pd_data, percent_var=95):
    """
    :pd_data: the dataframe containing the 'already standardized'] data 
    :percent_var: float - a value between [0,100]
    """
    # add checking if percent_var between 0 and 100

    # Calculate the desired number of SVD components in the decomposition
    start_rank = (pd_data.shape[-1]-1)
    # Make instance of SVD object class from scikit-learn and run the decomposition
    # Issue: scikitlearn TruncatedSVD only allows n_components < n_features (strictly)
    SVD = TruncatedSVD(n_components=start_rank)
    SVD.fit(pd_data)
    X_SVD = SVD.transform(pd_data)

    # Wrap the output as a dataframe
    X_SVD = pd.DataFrame(
        X_SVD, columns=['Singular Component '+str(i+1) for i in range(X_SVD.shape[-1])])

    # Calculate the number of components needed to reach variance threshold
    var_per_comp = SVD.explained_variance_ratio_

    # Calculate the total variance explainend in the first k components
    total_var = 100*np.cumsum(var_per_comp)
    print('------------- SVD Output ----------------')
    print('Percent Variance Explained By First ' +
          str(start_rank)+' Components: ', total_var, '\n\n')
    #rank = np.nonzero(total_var>=var_threshold)[0][0]+1
    rank = (next(x for x, val in enumerate(total_var) if val > percent_var))
    rank += 1

    if rank == 0:
        print('No quantity of components leq to '+str(start_rank+1) +
              ' can explain '+str(percent_var)+'% variance.')
    else:
        print(str(total_var[rank-1])+'% variance '+'explained by '+str(rank)+' components. ' +
              'Variance Threshold Was '+str(percent_var)+'.\n\n')

    return X_SVD, rank, percent_var, total_var


def data_quantization(pd_data, scale=10):
    """
    Quantize a panda data frame into integer with new features according to the given scale.
    e.g. if scale = 10: the new feature assign label 1 to the first, and 10 to the last
    :param pd_data:
    :param scale:
    :return: data_quantile: the quantized data
             percent_of_zero: at least that much percent of feature are zeros
    """
    p = np.linspace(0, scale, scale+1) * 0.1
    data_quantile = pd_data.copy()
    percent_of_zero = {}
    eps = 1e-5

    for feature in pd_data.columns:
        feature_new = feature + '_QUANTILE'
        data_quantile[feature_new] = 0

        for (i, quantile) in enumerate(p[:-1]):
            quantile_filter = np.quantile(
                pd_data[feature], [quantile, p[i + 1]])
            data_quantile.loc[((pd_data[feature] > quantile_filter[0]) &
                               (pd_data[feature] <= quantile_filter[1])), feature_new] = i+1

            # deal with 0-quantile being non-zero
            if i == 0 and quantile_filter[0] > 0:
                data_quantile.loc[((pd_data[feature] >= quantile_filter[0]) &
                                   (pd_data[feature] <= quantile_filter[1])), feature_new] = i+1

            if quantile_filter[0] <= eps and quantile_filter[1] >= eps:
                percent_of_zero[feature] = quantile
            elif quantile_filter[0] > eps and i == 0:
                percent_of_zero[feature] = 0

        data_quantile.drop(columns=feature, axis=1, inplace=True)
    return data_quantile, percent_of_zero


def get_elbow_index(scores):
    """
    Found online, idea is to draw the line segment between the starting score
    and the ending score and find the fartherest point to the line
    :param scores: SoS value
    :return: the index of elbow of the SoS scores assuming a uniform step size
    """
    curve = scores
    num_points = len(curve)
    allCoord = np.vstack((range(num_points), curve)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(
        vecFromFirst * np.matlib.repmat(lineVecNorm, num_points, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)

    return idxOfBestPoint


def plot_optimal(Xin, labels, num_comps=4, method='Kmeans', savepath=None, annotation = False, ft_select = None):
    """
    Paramters
        :Xin: <np.ndarray, shape (n,d)>  pca transformed data; rows in order of data matrix 
        :labels: <np.ndarray> labels from clustering algorithm for each row 
        :num_comps: <int> number of components to plot the PCA score for
        :method: <str> should be one of 'Kmeans' or 'GMM'
        :savepath: <str> the full filepath to where the image should be saved 
    Returns 
        None - void function
    """

    assert num_comps >= 2, 'Number of Components to Plot Must be at Least 2'
    assert (num_comps <= Xin.shape[-1]), 'Number of Components to \
        Plot must Be Lesser Than Dimensions of Data'

    clusters = list(set(labels.tolist()))
    clusters.sort()

    fig, axs = plt.subplots(1, num_comps-1)
    # Custom with used to see component value
    fig.set_size_inches(17.0, 8.0, forward=True)
    fig.subplots_adjust(wspace=.5)
    fig.suptitle('PCA View of {} Optimal Clustering'.format(method),
                 fontsize=22, fontname="Times New Roman", fontweight='bold')
    #cmap = plt.get_cmap('tab20')
    cmap = ListedColormap(['#bf5700','#005f86','#a6cd57','#333f48','#f8971f','#ffd600','#579d42','#00a9b7','#9cadb7','#d6d2c4','#f9ac4d','#ffe770','#cde3a1','#9acf8c','#99f7ff','#5cd1ff','#c4ced4'])
    idx = int(0)

    # Find way to wrap this in new iterable
    for cluster in clusters:
        idx = (idx+1) % 20
        Xcluster = Xin[cluster == labels,:]
        if annotation: #meaning doing the transpose problem
            Xname = np.array(ft_select)[np.where(cluster == labels)[0]]

        for comp in range(num_comps-1):
            ax = axs[comp]
            axs[comp].scatter(Xcluster[:, comp], Xcluster[:, comp+1],
                              color=cmap(idx), facecolor='none', label='Cluster '+str(cluster))
            ax.set_xlabel('Component '+str(comp+2), fontsize=14,
                      fontname="Times New Roman", fontweight='bold')
            ax.set_ylabel('Component '+str(comp+1),fontsize=14,
                      fontname="Times New Roman", fontweight='bold')

            if annotation:
                for (i, txt) in enumerate(Xname):
                    ax.annotate(txt, (Xcluster[i, comp], Xcluster[i, comp+1] ))
            else:
                pass


    # there is a way to attach the legend to figsave - find this when needed to plot
    # for ax in axs:
        # ax.legend()
    plt.show()
    
    pass

def elbow_method(X, k_search, method='KMeans', plot=True, savedir = './presimages', do_transpose = False, feature_select = None):
    """
    Elbow Method for different clustering methods with metrics CHindex, DBindex shown;
    Additionally show SoS for kMeans clustering. And finally plot the silhouette scores
    :param X: (np,ndarray NxD) data matrix 
    :param k_search: (np.ndarray) list containing the number of clusters to compare over
    :param method: (string) "Kmeans" or "GM" specify the clustering techniques
    :param plot: (boolean) if plotting the results or not
    :param do_transpose: (boolean) if this is the transpose problem
    :return: 
    """
    # ksearch must be linear for this to work
    k_diff = k_search[1:len(k_search)] - k_search[0:len(k_search) - 1]
    assert min(k_diff) == max(
        k_diff), "k_search does not have uniform increment"

    silh_score = np.zeros(len(k_search))
    CHindex_score = silh_score.copy()
    DBindex_score = silh_score.copy()
    SoS = silh_score.copy()

    if method == 'KMeans' or method == 'GM':
        pass
    else:
        raise ValueError(
            'method is not a valid method (only "kmeans" or "gm" is available)')

    # To plot a given clustering, we run PCA first and preserve ordering of rows. Then select indices
    # for each cluster label in order to plot

    print("Running Elbow Method...")
    for (i, k) in tqdm(enumerate(k_search), total=len(k_search)):
        if method == 'KMeans':
            kmeans = KMeans(n_clusters=int(k), random_state=0).fit(X)
            kmeans_label = kmeans.labels_
            SoS[i] = kmeans.inertia_
            silh_score[i] = metrics.silhouette_score(
                X, kmeans_label, metric='euclidean')
            CHindex_score[i] = metrics.calinski_harabasz_score(X, kmeans_label)
            DBindex_score[i] = metrics.davies_bouldin_score(X, kmeans_label)
            # plot_optimal(X,kmeans_label)

        elif method == 'GM':
            gm = GaussianMixture(n_components=int(k), random_state=0).fit(X)
            gm_label = gm.predict(X)
            silh_score[i] = metrics.silhouette_score(
                X, gm_label, metric='euclidean')
            CHindex_score[i] = metrics.calinski_harabasz_score(X, gm_label)
            DBindex_score[i] = metrics.davies_bouldin_score(X, gm_label)

    metric_list = [CHindex_score, DBindex_score, SoS]

    # silh_score / np.max(np.abs(silh_score))

    metric_legend = ['CHindex', 'DBindex', 'SoS']

    if plot:
        if method == 'KMeans':
            m = 3
        elif method == 'GM':
            m = 2

        Markers = ['+', 'o', '*', 'x']

        fig = plt.figure(figsize=(15, 5))
        cmap = ListedColormap(['#bf5700','#005f86','#a6cd57','#333f48','#f8971f','#ffd600','#579d42','#00a9b7','#9cadb7','#d6d2c4','#f9ac4d','#ffe770','#cde3a1','#9acf8c','#99f7ff','#5cd1ff','#c4ced4'])
        #cmap = plt.get_cmap("tab10")
        for i in range(m):
            ax = fig.add_subplot(1, m, i+1)
            ax.plot(k_search, metric_list[i], marker=Markers[i], color=cmap(i))
            ax.set_title('Score of {}'.format(
                metric_legend[i]), fontname="Times New Roman", fontweight='bold')
        fig.supxlabel(r'Number of clusters $k$', fontsize=20,
                      fontname="Times New Roman", fontweight='bold')
        fig.supylabel('Metric Score', fontsize=20,
                      fontname="Times New Roman", fontweight='bold')
        fig.suptitle('Evaluation of {} clustering'.format(method),
                     fontsize=22, fontname="Times New Roman", fontweight='bold')

        if method == 'KMeans':
            # get the optimal sum of squares elbow value
            elbow_index = get_elbow_index(metric_list[-1])
            optimal_K_1 = k_search[elbow_index]

            kn = KneeLocator(k_search, metric_list[-1],
                             curve='convex',
                             direction='decreasing',
                             interp_method='polynomial',)
            optimal_K_2 = kn.elbow

            print('The elbow (num of clusters) of SoS given by Method 1 is {}, by Method 2 is {}\n'.format(
                optimal_K_1, optimal_K_2), flush='True')
        fig.show()
        # formerly was plt.show()  - is an issue here?


        center = input(
            'Now input the center of the fine-search interval (+-4) of the silhouette scores, (min is 6, press Enter to pass) \n')

        if center == '':
            return

        # Compute the Silhouette and Plot

        print('Now computing silhouette over each cluster number from {} to {}'.format(
            int(center)-4, int(center)+4))
        silh_interval = np.arange(int(center)-4, int(center)+5)

        fig, axs = plt.subplots(3, 3, figsize=(
            15, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.35, wspace=.2)
        axs = axs.ravel()
        CHindex_score2 = np.zeros(9)
        DBindex_score2 = CHindex_score.copy()

        for j in tqdm(range(9)):
            num_cluster = silh_interval[j]

            if method == 'KMeans':
                cluster = KMeans(n_clusters=int(num_cluster),
                                 random_state=0).fit(X)
                cluster_label = cluster.labels_

            elif method == 'GM':
                cluster = GaussianMixture(n_components=int(
                    num_cluster), random_state=0).fit(X)
                cluster_label = cluster.predict(X)

            silh_avg_score = metrics.silhouette_score(
                X, cluster_label, metric='euclidean')
            # print('Num of Cluster is {}. Average Silhouette is {:.2f} \n'.format(num_cluster, silh_avg_score))
            sample_silhouette_values = silhouette_samples(X, cluster_label)
            CHindex_score2[j] = metrics.calinski_harabasz_score(X, cluster_label)
            DBindex_score2[j] = metrics.davies_bouldin_score(X, cluster_label)

            y_lower = 5
            for i in range(num_cluster):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_label == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                #color = matplotlib.cm.nipy_spectral(float(i) / num_cluster)
                utcolor = ['#bf5700','#005f86','#a6cd57','#333f48','#f8971f','#ffd600','#579d42','#00a9b7','#9cadb7','#d6d2c4','#f9ac4d','#ffe770','#cde3a1','#9acf8c','#99f7ff','#5cd1ff','#c4ced4']
                axs[j].fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    #facecolor=color,
                    #edgecolor=color,
                    facecolor=utcolor[i],
                    edgecolor=utcolor[i],
                    #alpha=0.7,
                    alpha=1,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                axs[j].text(-0.05, y_lower + 0.2 *
                            size_cluster_i, str(i), fontsize=6)

                # Compute the new y_lower for next plot
                y_lower = y_upper + 5  # 10 for the 0 samples

            axs[j].set_title('Num of Cluster: {}'.format(num_cluster))
            axs[j].set_xlabel('Avg Score = {:.3f}'.format(silh_avg_score))
            axs[j].axvline(x=silh_avg_score, color="red", linestyle="--")
            axs[j].set_yticks([])
        fig.supxlabel('Silhouette Score', fontsize=20,
                      fontname="Times New Roman", fontweight='bold')
        fig.supylabel('Cluster Label', fontsize=20,
                      fontname="Times New Roman", fontweight='bold')
        fig.suptitle('Silhouette Score for each sample', fontsize=22,
                     fontname="Times New Roman", fontweight='bold')
        plt.show()

    """
    ---------------- New ---------------
    Need for method to identify optimal clusterings 
    
    plot optimal clustering after decomposition of the data 
    ------------------------------------------------------
    """
    
    optimal_num = input('Input an integer value for optimal clusters based on inspection: ')
    optimal_num = int(optimal_num.strip())
    optimal_indx = int(np.where(silh_interval==optimal_num)[0])
    print("For this number of cluster, the CH score is {}, the DB score is {}".format(CHindex_score2[optimal_indx], DBindex_score2[optimal_indx]))
    
    print('Optimal input was',optimal_num)
    makePCA = PCA(n_components=np.min((X.shape[0]-1, X.shape[1]-1)))
    makePCA.fit(X)
    Xpca = makePCA.transform(X)
    # Re-run kmeans for optimal number

    if do_transpose:
        num_comps = 3
    else:
        num_comps = 5

    #assert (type(optimal_K_1) == int), 'First Optimal KMeans Cluster Value is Not Integer'
    #assert (type(optimal_K_2) == int), 'Second Optimal KMeans Cluster Value is Not Integer'
    if method == 'GM':
        GMMOpt = GaussianMixture(n_components = optimal_num, random_state=0).fit(X)
        optimal_label_gmm = GMMOpt.predict(X)
        plot_optimal(Xin = Xpca, labels= optimal_label_gmm, num_comps= num_comps,\
                     method = method, savepath = savedir+'/optimal_gmm.eps', annotation = do_transpose, ft_select = feature_select)
    elif method == 'KMeans':
        KmeansOpt = KMeans(n_clusters=optimal_num, random_state=0).fit(X)
        optimal_label_kmeans = KmeansOpt.labels_
        plot_optimal(Xin = Xpca, labels= optimal_label_kmeans, num_comps= num_comps,\
                     method = method, savepath = savedir+'/optimal_kmeans.eps', annotation = do_transpose, ft_select = feature_select)
            
    # previous way of showing figures
        # for i in range(m):
        #     plt.plot(k_search, metric_list[i], marker = Markers[i])
        # plt.xlabel(r'Number of clusters $k$', fontsize=20, fontname="Times New Roman", fontweight='bold')
        # plt.ylabel('Metric Score (Normalized)', fontsize=20, fontname="Times New Roman", fontweight='bold')
        # plt.title('Evaluation of {} clustering'.format(method), fontsize=22, fontweight='bold')
        # plt.legend(metric_legend, loc='best')
        # plt.show()

    return optimal_num


def plot_individual_feature(X, labels, num_cluster, feature_list, feature_plot):

    for i in range(num_cluster):
        ith_cluster_member = X[labels == i]
        nrows = np.ceil(np.sqrt(len(feature_plot)))
        ncols = np.ceil(len(feature_plot)/nrows)
        fig, axs = plt.subplots(int(nrows), int(ncols), figsize=(
            15, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.3) # suitable for 9 plots, need to adjust for other number of plots
        axs = axs.ravel()
        for j in range(len(feature_plot)):
            feature_index = feature_list.index(feature_plot[j])
            axs[j].hist(ith_cluster_member[:,feature_index], bins=10)
            axs[j].set_title('Feature {}'.format(feature_plot[j]), fontsize=14,
                      fontname="Times New Roman", fontweight='bold')
            axs[j].set_xlabel('Magnitude of the feature', fontsize=12,
                      fontname="Times New Roman", fontweight='bold')
            axs[j].set_ylabel('Number of Population', fontsize=12,
                      fontname="Times New Roman", fontweight='bold')
        fig.suptitle('{}th Cluster'.format((i)), fontsize=18,
                      fontname="Times New Roman", fontweight='bold')
        fig.savefig('images/Individual_Feature/SelectedFeature{}thCluster.png'.format(i))
        # plt.show()


if __name__ == "__main__":

    pass
