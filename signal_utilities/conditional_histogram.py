import numpy as np
import pdb
    
def compute_conditional_histogram(dep_array, indep_array, **kwargs):
    '''compute the conditional histogram i.e. if indep_array is large in magnitude, what is
    the probability that dep_array is large,small,etc.
    ''' 
    dep_min  = np.min(dep_array)
    dep_max  = np.max(dep_array)
    indep_min = np.min(indep_array)
    indep_max = np.max(indep_array)

    flagint  = np.sum(np.abs(dep_array-np.round(dep_array)))==0
    flagintc = np.sum(np.abs(indep_array-np.round(indep_array)))==0
    flagneg  = np.sum(dep_array<0)>0
    flagnegc = np.sum(dep_array<0)>0

    dep_bins = 21
    if 'dep_bins' not in kwargs:
        if flagint and flagneg:
            dep_max = np.max(np.abs(dep_array))
            dep_min = -dep_max
        if flagint:
            dep_bins = dep_max-dep_min+1
    else:
        dep_bins = kwargs['dep_bins']        
        
    indep_bins = 21
    if 'indep_bins' not in kwargs:
        if flagintc and flagnegc:
            indep_max = np.max(np.abs(indep_array))
            indep_min = -indep_max
        if flagintc:
            indep_bins = indep_max-indep_min+1
    else:
        indep_bins = kwargs['indep_bins']        

    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
    else:
        normalize = 1

    indep_bin_width = (indep_max-indep_min)/(indep_bins)
    dep_bin_width = (dep_max-dep_min)/(dep_bins)
    # indep_bin_range = np.arange(indep_min, indep_max, indep_bin_width)
    # dep_bin_range = np.arange(dep_min, dep_max, dep_bin_width)
    if 'log_scaling' in kwargs:
        indep_bin_range = np.logspace(np.log10(indep_min), np.log10(indep_max), num=indep_bins,endpoint=False)
        dep_bin_range = np.logspace(np.log10(dep_min), np.log10(dep_max), num=dep_bins,endpoint=False)
    else:
        indep_bin_range = np.linspace(indep_min, indep_max, num=indep_bins,endpoint=False)
        dep_bin_range = np.linspace(dep_min, dep_max, num=dep_bins,endpoint=False)

    print len(indep_bin_range)
    print len(dep_bin_range)    
    histogram = np.zeros([dep_bins, indep_bins])

    for indep_ix in xrange(len(indep_bin_range)):
        indep_bin_locs = np.abs(indep_array-(indep_bin_range[indep_ix] + indep_bin_width)) <= indep_bin_width / 2
        for dep_ix in xrange(len(dep_bin_range)):
            histogram[dep_ix,indep_ix] = np.sum(np.abs(dep_array[indep_bin_locs]-(dep_bin_range[dep_ix] + dep_bin_width)) <= dep_bin_width/2 )

    if normalize:
        histogram_normalizer = np.tile(np.sum(histogram,axis=0),(dep_bins, 1))
        eps = np.finfo(histogram_normalizer.dtype).eps
        histogram_normalizer[histogram_normalizer < eps] = eps
        histogram /= histogram_normalizer
        
    return dep_bin_range, indep_bin_range, histogram

def uniform_quantization(input_array, bins):
    '''perform uniform quantization (Q_bins(input_array)) on input_array
    Q_bins(input_array) = 0    if  |x|<T
     Q_bins(input_array) = sign(input_array) * ([input_array/bins]+0.5)*bins      where [.]=floor

    '''
    small_indices =  np.abs(input_array) < bins
    quantization_levels = np.floor(np.abs(input_array) / bins)
    return np.sign(input_array) * quantization_levels