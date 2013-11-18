#!/usr/bin/python -tt
"""
a collection of signal/matrix utility functions
"""
import numpy as np

def nd_impulse(ary_size):
    ary_impulse = np.zeros(ary_size)
    ary_impulse[tuple(np.array(ary_impulse.shape)/2)] = 1
    return ary_impulse

def spectral_radius():
    

def circshift(ary_input, tup_shifts):
    """Shift array circularly.
  
    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.
  
    Parameters
    ----------
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

