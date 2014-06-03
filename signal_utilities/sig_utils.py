#!/usr/bin/python -tt
from operator import add
import numpy as np
from numpy import max as nmax, absolute, conj, arange, zeros, array, median, pad
from numpy.fft import fftn, ifftn
from numpy.linalg import norm
from numpy.random import normal, rand, seed, poisson
import itertools

def nd_impulse(ary_size):
    ary_impulse = zeros(ary_size)
    ary_impulse[tuple(array(ary_impulse.shape)/2)] = 1
    return ary_impulse

def spectral_radius(op_transform, op_modality, tup_size, method='spectrum'):
    """Compute the specral radius (the dominant eigenvalues) of the composition of op_transform and op_modality.
    :param op_transform: a transform operator which returns a ws object
    :param op_modality: some linear operator, 
     assumes this is diagonalizable by the fft (e.g. convolution)
    :param tup_size: size of impulse used...
    :param method: the available methods for computing these weights (spectrum or power_iteration)
    :returns ary_lambda_alpha, a vector of subband weights which upperbounds the spectral radius
    
    .. codeauthor:: Timothy Roberts <timothy.daniel.roberts@gmail.com>, 2013
    """
    if method=='spectrum':
        ary_impulse = nd_impulse(tup_size)
        ws_w = op_transform * ary_impulse
        ary_alpha = zeros(ws_w.int_subbands)
        ary_temp = op_modality * ary_impulse # unused result, just to initialize with correct size
        ary_ms = op_modality.get_spectrum()
        if tup_size != ary_ms.shape:
            ary_impulse = nd_impulse(ary_ms.shape)
            ws_w = op_transform * ary_impulse
        ary_ms = ary_ms.flatten()
        ary_subbands = arange(ws_w.int_subbands)
        ary_ss = zeros([ary_ms.shape[0],ws_w.int_subbands],dtype='complex64')
        ary_alpha = zeros(ws_w.int_subbands,)
        #store the inband psd's (and conjugate)
        for s in ary_subbands:
            ary_ss[:,s] = fftn(~op_transform * ws_w.one_subband(s)).flatten()
        #compute the inband and crossband psd maxima
        for s in ary_subbands:
            for c in ary_subbands:
                 ary_alpha[s] += nmax(absolute(conj(ary_ms * ary_ss[:,s]) * ary_ms * ary_ss[:,c]))
        #return the alpha array, upper-bounded to 1                      
        ary_alpha = np.minimum(ary_alpha,1.0)
        
    elif method=='power_iteration':
        W = op_transform
        Phi = op_modality
        ws_rv = op_transform * rand(*tup_size) #random vector
        ary_eigs = np.zeros(ws_rv.int_subbands)
        for s in xrange(ws_rv.int_subbands):
            ws_ss = ws_rv.one_subband(s)
            i = 0
            while i < 30:
                ws_ss = A(W,Phi,ws_ss) #to image domain
                ws_ss = At(W,Phi,ws_ss) #back to wavelet domain
                # pdb.set_trace()
                ws_ss_norm = norm(ws_ss.flatten(),2)
                for s2 in xrange(ws_ss.int_subbands):
                    ws_ss.set_subband(s2,ws_ss.get_subband(s2)/ws_ss_norm)
                i += 1    
                ws_ss = ws_ss.one_subband(s)    
            AtAws_ss = A(W,Phi,ws_ss)
            AtAws_ss = At(W,Phi,AtAws_ss).flatten()
            print np.sum(ws_ss.flatten().transpose() * AtAws_ss)
            print ws_ss_norm
            ary_eigs[s] = np.sum(ws_ss.flatten().transpose() * AtAws_ss) / ws_ss_norm
            print ary_eigs[s]
            print ary_eigs
        ary_alpha = np.minimum(ary_eigs,1.0)
    else:
        raise ValueError('method ' + method + ' unsupported')
                
    print ary_alpha
    return ary_alpha


def A(op_transform,op_modality,mcand):
    return op_modality * (~op_transform * mcand)

def At(op_transform,op_modality,mcand):
    return op_transform * (~op_modality * mcand)


def circshift(ary_input, tup_shifts):
    """Shift multi-dimensional array circularly.
  
    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.
  
    Parameters
    
    ary_input : ndarray to shift.
  
    s : tuple of int
       A tuple of integer scalars where the N-th element specifies the
       shift amount for the N-th dimension of input array. If an element
       is positive, the values of `a` are shifted down (or to the
       right). If it is negative, the values of `a` are shifted up (or
       to the left).

    Returns
    -------
    y : ndarray
       The shifted array (elements are copied)
  
    Examples
    --------
    >>> circshift(np.arange(10), 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
  
    """
    # Initialize array of indices
    idx = []
  
    # Loop through each dimension of the input matrix to calculate
    # shifted indices
    for dim in range(ary_input.ndim):
        length = ary_input.shape[dim]
        try:
            shift = tup_shifts[dim]
        except IndexError:
            shift = 0  # no shift if not specify
  
        index = np.mod(np.array(range(length),
                                ndmin=ary_input.ndim) - shift,
                                length)

        shape = np.ones(ary_input.ndim,dtype='uint16')
        shape[dim] = ary_input.shape[dim]
        index = np.reshape(index, shape)
        idx.append(index.astype(int))
  
    # Perform the actual conversion by indexing into the input matrix
    return ary_input[idx]

def noise_gen(noise_params):
    '''
    A funtion to compute realizations of an additive noise process
    Expects (in noise_params dictionary):
    seed, variance, distribution, mean, size, (interval, for noise with finite support pdf)
    '''
    int_seed = noise_params['seed']
    dbl_mean = noise_params['mean']
    dbl_variance = noise_params['variance']
    tup_size = noise_params['size']

    #set the seed, always set the seed!
    seed(int_seed)
    if noise_params['distribution'] == 'gaussian':
        ary_noise = normal(dbl_mean, np.sqrt(dbl_variance), tup_size)
    elif noise_params['distribution'] == 'uniform':
        ary_interval = noise_params['interval']
        ary_noise = (ary_interval[1] - ary_interval[0]) * rand(tup_size) + ary_interval[0]
    elif noise_params['distribution'] == 'poisson':     
        ary_noise = poisson(lam=noise_params['ary_mean'])
    else:
        raise Exception('unsupported noise distribution: ' + noise_params['distribution'])
    return ary_noise

def colonvec(ary_small, ary_large):
    """
    Compute the indices used to pad/crop the results of applying the fft with augmented dimensions
    ary_small and ary_large are shape tuples which define the bounds of the cropping region
    """
    int_max = np.maximum(len(ary_small),len(ary_large))
    indices = [np.s_[int(ary_small[i]-1):int(ary_large[i])] for i in np.arange(int_max)]
    return indices

def pad_center(ary_input, tup_new_size):
    """
    Returns a new array of size tup_new_size with ary_input in the center
    """
    if ((ary_input.shape > tup_new_size) or 
        (ary_input.ndim != len(tup_new_size))): #one of the dimensions is too big
        raise ValueError('cannot pad ary_input using new shape ' + 
                         str(tup_new_size) + ', input shape is ' + 
                         ary_input.shape)
    else: #inputs are OK
        pad_sz=(np.array(tup_new_size)-np.array(ary_input.shape))/2
        ls_pad_size_axis=[(pad_sz[j],pad_sz[j]) for j in xrange(pad_sz.size)]
        return np.pad(ary_input,ls_pad_size_axis,mode='constant',constant_values=0)

def get_neighborhoods(ary_input,n_size):
    """
    Given an array, return the neighborhoods of size n_size along a new axis.

    The return value is a tuple of values and a logical mask. The values are the elements of the neighborhood,
    and the mask tells you which (since not all neighborhoods are the same size) elements are valid

    Here, x's show a neighborhood of coefficient 'o' size 1 for a 2-d array:
    a b c 
    d o e
    f g h

    The function would return 9 arrays of the same size of 1. values and 2. masks. Here is the first of 1 arrays
    value:
    o e d 
    g h f
    b c a
    mask:
    True True False
    True True False
    False False False

    This makes it convenient for performing vector operations (e.g. average) on the neighborhoods, which is the whole reason for writing this function.
    
    """
    #calculate shifts
    dims = ary_input.ndim
    iter_dims = xrange(dims)
    grid = np.mgrid[[slice(-n_size,n_size+1 ,None) for i in iter_dims]]
    shifts = np.vstack([grid[i,...].flatten() for i in iter_dims]).transpose()
    iter_shifts = xrange(len(shifts))
    neighborhoods = [circshift(ary_input,tuple(shifts[i])) for i in iter_shifts]
    #now we have to zeroize the rows,columns,etc where edges have been shifted in
    for i in iter_shifts:
        ls_slices = shift_slices(shifts[i])
        if ls_slices!=[]:
            for j in xrange(len(ls_slices)):             
                neighborhoods[i][ls_slices[j]]=np.nan
    neighborhoods = np.concatenate([neighborhoods[i][...,np.newaxis] 
                                    for i in iter_shifts],axis=dims)
    mask = ~np.isnan(neighborhoods)
    neighborhoods[~mask]=0
    return tuple([neighborhoods,mask])

def shift_slices(shift):
    """calculate the slices corresponding to a given set of shifts
    e.g. shift=(-1,-1) would give a list of lists of slices : [[slice(-1,None,None),slice(None,None,None)],
                                                              [slice(None,None,None),slice(-1,None,None)]]
    """
    ls_slices = []
    iter_shifts = xrange(len(shift))
    if np.all(shift==0): #just return empty list for indexing nothing
        return []
    eslice = slice(None,None,None)
    for k in iter_shifts:
        if shift[k]!=0:
            slices = []
            for j in iter_shifts:
                if j==k:
                    if shift[k]<0:
                        sl=slice(shift[k],None,None)
                    else: #shift[k]>0:    
                        sl=slice(None,shift[k],None)
                else:
                    sl=eslice        
                slices.append(sl)
            ls_slices.append(slices)
    return ls_slices            

def downsample_slices(int_dim):
    """Return a list of list of slices used for downsampling by a factor of 2
    in every dimension, for each of 2**int_dim offsets
    """
    num_slices = 2**int_dim
    return [[slice(dec_to_bin(j,int_dim)[i],None,2) for i in xrange(int_dim)] for j in xrange(num_slices)]
            
def dec_to_bin(int_decimal,width=None):
    """returns int_decimal as a list of binary digits, with the lsbit
    being the first element, with msb padding
    """ 
    bin_string = bin(int_decimal)
    bin_width = len(bin_string)-2
    if width == None:
        width=bin_width
    return [int(bin_string[i]) for i in xrange(bin_width+1,1,-1)] + (width-bin_width)*[0] 
    
def crop_center(ary_signal, tup_crop_size):
    '''Crop ary_signal symmetrically to be tup_crop_size
    '''
    ary_half_difference = (array(ary_signal.shape) - array(tup_crop_size)) / 2
    return ary_signal[colonvec(ary_half_difference+1, ary_half_difference+array(tup_crop_size))]

def gaussian(shape=(3,3),sigma=(0.5,0.5)):
    """
    2D gaussian - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma[0]*sigma[1]) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def upsample(ary_input,factor=2,method='shiftadd'):
    ary_upsampled = np.zeros(np.array(ary_input.shape)*factor)
    if method=='shiftadd':
        if ary_upsampled.ndim!=2:
            raise ValueError('dimensions other than 2 not supported')
        y1=np.zeros((ary_upsampled.shape[0],ary_input.shape[1]))
        y1[slice(0,-1,2),...] = (0.75*ary_input 
                                    + 0.25*np.concatenate(
                                        [ary_input[slice(0,1),...], 
                                         ary_input[slice(0,-1),...]],axis=0))
        y1[slice(1,None,2),...] = (0.75*ary_input 
                                      + 0.25*np.concatenate(
                                            [ary_input[slice(1,None),...],
                                             ary_input[slice(-1,None),...]],axis=0))
        ary_upsampled[...,slice(0,-1,2)] = (0.75*y1 
                                               + 0.25*np.concatenate(
                                                   [y1[...,slice(0,1)],
                                                    y1[...,slice(0,-1)]],axis=1))
        ary_upsampled[...,slice(1,None,2)] = (0.75*y1
                                               + 0.25*np.concatenate(
                                                   [y1[...,slice(1,None)],
                                                   y1[...,slice(-1,None)]],axis=1))
        return ary_upsampled
    else:
        raise ValueError('unsupported upsample method ' + method)
    
def flatten_list(ls_ary):
    '''Flatten a list of objects which have the flatten() method
    Assumes each element is the same size
    '''
    temp_ary=ls_ary[0].flatten()
    ary_size=temp_ary.size
    vec_ix=0
    output_ary=np.zeros(ary_size*len(ls_ary))
    for ary_unflat in ls_ary:
        output_ary[vec_ix:vec_ix+ary_size]=ary_unflat.flatten()
        vec_ix+=ary_size
    return output_ary

def unflatten_list(ary_input,num_partitions):
    '''Split ary_input into num_partitions equal-sized arrays and store 
    these in a list
    '''
    int_part_size=ary_input.size/num_partitions
    ls_ary=[]
    vec_ix=0
    for partition in xrange(num_partitions):
        ls_ary.append(ary_input[vec_ix:vec_ix+int_part_size])
        vec_ix+=int_part_size
    return ls_ary

def inv_block_diag(csr_block_diag_mtx):
    '''input is a sparse block diagonal matrix
    ouput is the inverted version of this matrix
    '''
    csr_rows=np.nonzero(csr_block_diag_mtx)
    csr_cols=csr_rows[1]
    csr_rows=csr_rows[0]
    #scan through the rows, which should be sorted
    
    

def mad(data, axis=None):
    return median(absolute(data - median(data, axis)), axis)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])