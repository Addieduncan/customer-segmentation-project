





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
    bad_rows = list()
    
    for ft in features:
        feature_series = pd_data[ft]
        missing_bool = feature_series.isnull()
        bad_indices = feature_series.index[missing_bool]
        #Calculate the percentage of that feature which was True under .isnull()
        missing_dict[ft] = 100*float(np.sum(missing_bool)/feature_series.shape[0])
        
        if not bad_indices.empty:
            if verbose:
                print("Issue Feature:\n", ft,'\n', bad_indices, '\n Num of null=', len(bad_indices), '\n\n')
                bad_rows += list(bad_indices)
                print('Here are Nan Indices:', bad_indices)
            else:
                pass
            
    #Total percentage(s) of data removed
    if verbose:
        print('Number of Removed Row Instances = \n',bad_rows,'\n ')
        print('Percentage of Removed Features = \n',missing_dict)
    #Eliminate duplicates and sort 
    bad_rows = list(set(bad_rows))
    bad_rows.sort()

    # Get rid of rows containing null or empty
    clean_data = pd_data.drop(bad_rows)

    #Check the number of resulting data points 
    if verbose:
        print('Here is shape of original data:',data.shape,'\n\n')
        print('Here is shape of the clean data:', data_clean.shape,\
              '\n Number of Removed Instances =',len(bad_rows))
        
    return clean_data, missing_dict, bad_rows

def select_features(pd_data,which = 'basic'):
    """
    pd_data: should be the raw, completely unprocessed feature data 
    
    which:: <string> - determine which set of features to use 
    """
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
    if which == 'freq':
        ft_keep = ft_freq[:]
    elif which == 'all':
        ft_keep = ft_basic[:] + ft_freq[:]

    #Here are the features haven't been used
    ft_unused = set(list(features))-set(ft_keep)

    print('Here are the features related to $$:\n', ft_basic,'\n')
    print('Here are the features related to frequency: \n',ft_freq,'\n')
    print('Here are the features not used: \n', ft_unused)

    
    if which == 'basic':
        
        
        
        
        

if __name__ == "__main__":

	pass 