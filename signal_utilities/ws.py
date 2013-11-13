#!/usr/bin/python -tt
import numpy as np
class WS(object):
    """
    WS class for storing and retreiving and performing operations on wavelet subbands. 
    """
    
    def __init__(self,ary_scaling,tup_coeffs):
        """
        Class constructor for DTCWT
        """
        self.ary_scaling = ary_scaling
        self.tup_coeffs = tup_coeffs
        self.ary_size = np.dot(2,ary_scaling.shape) 
        self.int_levels = len(tup_coeffs)
        self.int_dimension = ary_scaling.ndim
        self.int_orientations = tup_coeffs[0].shape[-1]