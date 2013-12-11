#!/usr/bin/python -tt
from operator import add
import numpy as np
from numpy import max as nmax, absolute, conj, arange, zeros, array
from numpy.fft import fftn, ifftn
from numpy.random import normal, rand, seed
import itertools

def nd_impulse(ary_size):
    ary_impulse = zeros(ary_size)
    ary_impulse[tuple(array(ary_impulse.shape)/2)] = 1
    return ary_impulse

def spectral_radius(op_transform, op_modality, tup_size):
    """Compute the specral radius including crossband energy
    :param op_transform: a transform operator which returns a ws object
    :param op_modality: some linear operator, 
     assumes this is diagonalizable by the fft (e.g. convolution)

    :returns ary_lambda_alpha, a vector of subband weights which upperbounds the spectral radius
    
    .. codeauthor:: Timothy Roberts <timothy.daniel.roberts@gmail.com>, 2013
    """
    ary_impulse = nd_impulse(tup_size)
    ws_w = op_transform * ary_impulse
    ary_alpha = zeros(ws_w.int_subbands)
    ary_temp = op_modality * ary_impulse # unused result, just to initialize with correct size
    ary_ms = (op_modality.get_spectrum()).flatten()
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
    ary_alpha = np.minimum(ary_alpha,1)
    print ary_alpha
    return ary_alpha

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

        shape = np.ones(ary_input.ndim)
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
    print int_seed
    if noise_params['distribution'] == 'gaussian':
        ary_noise = normal(dbl_mean, np.sqrt(dbl_variance), tup_size)
    elif noise_params['distribution'] == 'uniform':
         ary_interval = noise_params['interval']
         ary_noise = (ary_interval[1] - ary_interval[0]) * rand(tup_size) + ary_interval[0]
    else:
        raise Exception('unsupported noise distribution: ' + noise_params['distribution'])
    return ary_noise