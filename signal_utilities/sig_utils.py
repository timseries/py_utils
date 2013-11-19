#!/usr/bin/python -tt
"""
a collection of signal/matrix utility functions
"""
from operator import add
import numpy as np
from numpy import max, absolute
import itertools

def nd_impulse(ary_size):
    ary_impulse = np.zeros(ary_size)
    ary_impulse[tuple(np.array(ary_impulse.shape)/2)] = 1
    return ary_impulse

def spectral_radius(op_transform, op_modality):
    """Compute the specral radius including crossband energy
    :param op_transform: a transform operator which returns a ws object
    :param op_modality: some linear operator, 
     assumes this is diagonalizable by the fft (e.g. convolution)

    :returns ary_lambda_alpha, a vector of subband weights which upperbounds the spectral radius
    
    .. codeauthor:: Timothy Roberts <timothy.daniel.roberts@gmail.com>, 2013
    """
    ary_impulse = nd_impulse(op_transform.ary_size)
    ls_sp = []
    ls_sp_conj = []
    ws_w = op_transform * ary_impulse
    ary_alpha = np.zeros(ws_w.int_subbands)
    ary_ms = op_modality.get_spectrum
    ary_ms_conj = np.conj(ary_ms)
    ary_subbands = np.arange(ws_w.int_subbands)
    #store the inband psd's (and conjugate)
    for int_subband in ary_subbands:
        ary_subband_spectrum = np.fftn(~op_transform * \
                                          ws_w.suppress_other_subbands(int_subband))
        ls_sp.append(ary_ms * ls_subband_spectra[int_subband])
        ls_sp_conj.append(ary_ms_conj *ls_subband_spectra[int_subband])
    #compute the inband and crossband psd maxima
    ls_ic = np.array([max(absolute(ls_sp[int_j] * ls_sp[int_n])) for \
                      int_j in ary_subbands for int_n in ary_subbands])
    ary_alpha = np.array([reduce(add,ls_ic[i:i+ary_subbands[-1]]) for \
                          i in (ary_subbands*ary_subbands[-1]+1)])
    #return the alpha array, upper-bounded to 1                      
    return np.minimum(ary_alpha,1)

def circshift(ary_input, tup_shifts):
    """Shift array circularly.
  
    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.
  
    Parameters!23Gamma
    
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

