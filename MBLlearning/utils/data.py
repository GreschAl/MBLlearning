##################################################################################################
# Contains functions used for data loading and preprocessing
# as well as convenience functions for obtaining their statistics
##################################################################################################

import numpy as np
from matplotlib.pyplot import plot,show

######################## data preprocessing ##################

def Paige_halfchain_entropy(L):
    """ Calculate average entropy of a locally-connected subspace (Paige entropy) """
    n = 2**(L//2)
    summand = 0.0
    for i in range(n+1,n**2+1):
        summand += 1/i
    summand /= np.log(2)
    summand -= (n-1)/(2*n)
    return summand

def preprocess_all_indicators(indic,L,keys):
    """ Normalization procedure on provided indicators with keys.
        Ensures that all indicator lie in the intervall [-1,1]
    """
    for i,key in enumerate(keys):
        # normalize adj. gap ratio r?
        
        # normalize KL div
        if key == "KLdiv":
            indic[:,i] -= 1.9
            KL_norm = 0.9/np.max(indic[:,i])
            indic[:,i] *= KL_norm
        
        # normalize Shannon entropy
        if key == "Shannon":
            indic[:,i] /= L*np.log(2)

        # normalize entanglement entropy
        if key == "EntEntropy":
            S_P = Paige_halfchain_entropy(L)
            indic[:,i] /= S_P

        # reshape S_corr to mostly be positive
        if key == "S_corr" and (L-6)%4 == 0:
            indic[:,i] *= -1
    return indic

######################## data loading ##################

def shuffle_seeded(a,seed=None):
    """ Shuffle the array a along first axis where the shuffling follows the given seed.
        This provides a compatible shuffling among multiple calls with the same seed.
        If no seed is provided, this functions reduces to np.random.shuffle()
    """
    if seed is None:
        np.random.shuffle(a)
    else:
        rs = np.random.RandomState(seed)
        rs.shuffle(a)
    return

def load_training_data(filename,preprocessed=True,no_params=3,test_data_only=False,
                       N_train=None,energies=None,single_inds=None,sorting=False,seed=None):
    """
    Load test (and train) data form a file location. Provide filelocation of train data only
    Options:
    - preprocessed (bool): apply normalization procedure (above)
    - no_params (int): provide number of parameter lines in data file
    - test_data_only (bool): if True, only load and return the test data only
    - N_train (None or float): if provided, use only N_train number of training samples for a given disorder parameter h.
        Otherwise, use all available train data
    - energies (list or comparable): if provided, obtain data from all energy regimes specified here and
      append the indicator values. In this case, filename should be a format string with a wildcard for the energy value.
    - single_inds (list): if provided, use only the corresponding indicator values as target values
    - sorting (bool): if energy values are provided, specify whether indicator values shall be stacked
        energy-wise or indicator-wise (default 'False' is energy-wise and corresponds to mere stacking of data)
    """
    if energies is None:
        if not test_data_only:
            # get data parameters from file first
            params = {}
            with open(filename,'r') as f:
                for i in range(no_params):
                    strip = f.readline().split()
                    params[strip[0]] = int(strip[-1])
                keys = f.readline().split()
                

            L = params['L']
            
            no_inds = len(keys) - 2 # count the number of target values
            
            if single_inds is not None:
                temp_keys = keys[:2]
                indicator_indices = [i for i in range(L+1)]
                for ind in single_inds:
                    assert ind in range(no_inds), "Chosen indicator index {} does not exist, only {} indicators are to choose from".format(ind, no_inds)
                    temp_keys.append(keys[ind+2])
                    indicator_indices.append(ind+L+1)
                no_inds = len(single_inds)
                keys = temp_keys # overwrite available indicator keys
            else:
                indicator_indices = None
                
            params['indicators'] = keys

            # load actual data afterwards
            data_train = np.loadtxt(filename,skiprows=no_params+1,usecols=indicator_indices)
            # if provided, subsample randomly
            if N_train is not None:
                assert N_train > 0, "N_train has to be chosen positive, but was {}".format(N_train)
                hcorr = data_train[:,0]
                h = np.unique(hcorr)
                temp = []
                for count,i in enumerate(h):
                    mask = hcorr == i
                    masked = data_train[mask]
                    if seed is not None:
                        temp_seed = seed+count+1
                        if temp_seed > 2**32: temp_seed -= 2*32
                    else:
                        temp_seed = None
                    shuffle_seeded(masked,temp_seed)
                    # each subsampling should be done independently of each other
                    temp.append(masked[:int(N_train)])
                data_train = np.concatenate(temp)
                print("Using {} per h-value (from initially set {})".format(len(temp[-1]),int(N_train)))
            # shuffle data for each loading procedure following the seed if provided
            # else, shuffle at random
            shuffle_seeded(data_train,seed)

            hcorr_train = data_train[:,0]
            hvals_train = data_train[:,1:L+1]
            inds_train  = data_train[:,L+1:]
            if no_inds == 1: inds_train = inds_train.reshape(-1,1)

        filename = filename.replace("train","test",1)

        if test_data_only:
            params = {}
            with open(filename,'r') as f:
                for i in range(no_params):
                    strip = f.readline().split()
                    params[strip[0]] = int(strip[-1])
                keys = f.readline().split()

            L = params['L']
            
            no_inds = len(keys) - 2 # count the number of target values
            
            if single_inds is not None:
                temp_keys = keys[:2]
                indicator_indices = [i for i in range(L+1)]
                for ind in single_inds:
                    assert ind in range(no_inds), "Chosen indicator index {} does not exist, only {} indicators are to choose from".format(ind, no_inds)
                    temp_keys.append(keys[ind])
                    indicator_indices.append(ind+L+1)
                no_inds = len(single_inds)
                keys = temp_keys # overwrite available indicator keys
            else:
                indicator_indices = None
                
            params['indicators'] = keys

        data_test   = np.loadtxt(filename,skiprows=no_params+1,usecols=indicator_indices)
        hcorr_test  = data_test[:,0]
        hvals_test  = data_test[:,1:L+1]
        inds_test   = data_test[:,L+1:]
        if no_inds == 1: inds_test = inds_test.reshape(-1,1)

        if preprocessed:
            # if this flag is activated (by default)
            if not test_data_only:
                # if this flag is activated (by default), transform the data according to the function above
                inds_train = preprocess_all_indicators(inds_train,L,keys[2:])
                inds_test  = preprocess_all_indicators(inds_test,L,keys[2:])
                return hcorr_train, hvals_train, inds_train, hcorr_test, hvals_test, inds_test, no_inds, params
            inds_test  = preprocess_all_indicators(inds_test,L,keys[2:])
        if test_data_only:
            return hcorr_test, hvals_test, inds_test, no_inds, params
        else:
            return hcorr_train, hvals_train, inds_train, hcorr_test, hvals_test, inds_test, no_inds, params
            
    else:
        # if energies are provided, use this function recursively to call itself to load the indicators energy-wise
        # and append them to each other in the same order as of energies
        no_eps = len(energies)
        
        if test_data_only:
            j_ = 0
            hcorr_test, hvals_test, inds_test, no_inds, params = load_training_data(filename.format(energies[0]),
                                                                                 preprocessed,no_params,True,N_train,
                                                                                   None,single_inds)
            if sorting:
                inds_test_full = np.zeros((len(hcorr_test),no_inds*no_eps))
                for k in range(no_inds):
                    inds_test_full[:,no_eps*k] = inds_test[:,k]
            
            for e in energies[1:]:
                j_ += 1
                fileloc = filename.format(e)
                _, _, inds, _, _ = load_training_data(filename.format(e),preprocessed,no_params,True,N_train,None,single_inds)
                # append new indicator values
                if sorting:
                    for k in range(no_inds):
                        inds_test_full[:,no_eps*k + j_] = inds[:,k]
                else:
                    inds_test = np.hstack((inds_test,inds))
            if not sorting:
                inds_test_full = inds_test # overwrite for compability for the two sorting methods
            return hcorr_test, hvals_test, inds_test_full, no_inds, params
        
        else:
            # fix a seed for the shuffling of the training data and use the same key for all energies
            seed = np.random.randint(2**32)
            
            j_ = 0
            hcorr_train, hvals_train, inds_train, hcorr_test, hvals_test, inds_test, no_inds, params = load_training_data(filename.format(energies[0]),preprocessed,no_params,False,N_train,None,single_inds,seed=seed)
            
            if sorting:
                inds_test_full  = np.zeros((len(hcorr_test),no_inds*no_eps))
                inds_train_full = np.zeros((len(hcorr_train),no_inds*no_eps))
                for k in range(no_inds):
                    inds_test_full[:,no_eps*k]  = inds_test[:,k]
                    inds_train_full[:,no_eps*k] = inds_train[:,k]
            
            for e in energies[1:]:
                j_ += 1
                _, hvals_train, train, _, hvals_test, test, _, _ = load_training_data(filename.format(e),preprocessed,no_params,
                                                                   False,N_train,None,single_inds,seed=seed)
                
                # append new indicator values
                if sorting:
                    for k in range(no_inds):
                        inds_test_full[:,no_eps*k + j_]  = test[:,k]
                        inds_train_full[:,no_eps*k + j_] = train[:,k]
                else:
                    inds_test  = np.hstack((inds_test,test))
                    inds_train = np.hstack((inds_train,train))
            if not sorting:
                inds_test_full = inds_test # overwrite for compability for the two sorting methods
                inds_train_full = inds_train
            return hcorr_train, hvals_train, inds_train_full, hcorr_test, hvals_test, inds_test_full, no_inds, params
            

######################## convenience functions for data statistics ##################

def _get_statistic(data,key,sorting_array,func):
    """ Helper function to apply the given function func along the first axis to data having matching features
        of elements in sorting_array. The features are given in the key array.
        
        Returns a list of length == len(sorting_array) with the function element-wise applied.
    """
    return [func(data[key==value],axis=0) for value in sorting_array]

def get_avg(model,L=None,key="test",eps_idx=None):
    """ Returns means and std dev. of disorder realization of train or test data, disorder parameter-wise
        held by a model.
        If L is provided (has to be for models with a variable chain length only), this is done for the
        provided chain length L
        If various energy values are used in a concatenated way (indicator- or energy-wise, specified by sorting), either
        obtain data for a single energy index (if eps_idx is provided) or for all returned as an array
    """
    data = model.data[key] if L is None else model.data[L][key]
    n = model.N_inds
    no_energies = None if model.energies is None else len(model.energies)
    sorting = model.sorting

    h = data["h"]
    hcorr = data["hcorr"]
    if no_energies is None:
        # vanilla version, get statistics over all "inds"-values
        indicators = data["inds"]
        
        means = _get_statistic(indicators,hcorr,h,np.mean)
        stds  = _get_statistic(indicators,hcorr,h,np.std)
    else:
        if eps_idx is None:
            # obtain statistics for each energy value and append in a list
            means = []
            stds = []
            for i in range(no_energies):
                if sorting:
                    indicators = data["inds"][:,i+np.arange(n)*no_energies]
                else:
                    indicators = data["inds"][:,i*n:(i+1)*n]
                means.append( np.array(_get_statistic(indicators,hcorr,h,np.mean)) )
                stds.append(  np.array(_get_statistic(indicators,hcorr,h,np.std))  )
        else:
            # obtain statistics for a single given energy value only
            if sorting:
                indicators = data["inds"][:,eps_idx+np.arange(n)*no_energies]
            else:
                indicators = data["inds"][:,eps_idx*n:(eps_idx+1)*n]
            means = _get_statistic(indicators,hcorr,h,np.mean)
            stds  = _get_statistic(indicators,hcorr,h,np.std)
                 
    return h,np.array(means),np.array(stds),len(hcorr)//len(h)

def get_sample(model,N,L=None,key="test"):
    """ Obtain N samples from the data held by a model
        If L is provided (has to be for models with a variable chain length only), this is done for the
        provided chain length L
    """
    data = model.data[key] if L is None else model.data[L][key]
    
    h = data["h"]
    hcorr = data["hcorr"]
    sample_inds = np.random.choice(len(hcorr),N) # obtain N sample indices
    hcorr = hcorr[sample_inds]
    inds  = data["inds"][sample_inds]
    h_i = data["h_i"][sample_inds]
    
    return h, hcorr, h_i, inds

def sort_by_h(data,hcorr,h):
    """ Given some unsorted data, sort it by their respective h-value by copying into a new array.
        Provided data is sorted over the first axis.
    """
    out = []
    for hval in h:
        mask = hcorr == hval
        out.append(data[mask])    
    return np.array(out)

def coefficient_of_determination(model,L=None,reduce_energies=False,key="test",transfer_ind=None):
    """ Obtain quantifier R^2 = 1 - MSE(prediction,target)/Var(target) = 1 - Var(prediction-target)/Var(target)
        for the model for the approximation accuracy.
        If reduce_energies is set to False (default), computes R in dependence of energy and disorder paramter h.
        Obsolete if model does not have any energy regimes.
        If a transfer_ind is provided (not by default), computes R only for this provided indicator value alone.
    """
    data = model.data if L is None else model.data[L]
    if data["estimation"] is None:
        # perform the estimation step if not previously done
        model.estimate(key)
    else:
        # check for right key in estimation data
        if data["estimation"]["key"] != key:
            model.estimate(key)
    
    h = data[key]["h"]
    hcorr = data[key]["hcorr"]
    indicators = sort_by_h(data[key]["inds"],hcorr,h)
    if transfer_ind is not None:
        assert isinstance(transfer_ind,int), "transfer_ind has to be an integer or None."
        indicators = ((model.sort(indicators))[...,transfer_ind]).transpose([1,2,0])

    # take the difference of estimation to the indicators
    estim_err  = (sort_by_h(data["estimation"]["inds"],hcorr,h) - indicators)**2
    
    if not reduce_energies or model.energies is None:
        # vanilla version, get statistics over all "inds"-values        
        var = np.var(indicators, axis=1)
        mse = np.mean(estim_err, axis=1)
    else:
        # obtain statistics for each energy value and append in a list
        # sort by energy using the model sorting function
        if transfer_ind is None:                
            estim_err  = model.sort(estim_err)
            indicators = model.sort(indicators)
            axis = 2
        else:
            axis = 1
        var = np.var(indicators, axis=axis)
        mse = np.mean(estim_err, axis=axis)
    return 1 - mse/var
