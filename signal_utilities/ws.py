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
        self.int_subbands = self.int_levels * self.int_orientations + 1

    def suppress_other_subbands(self,int_subband_index):
        """
        For a given subband index, fetches the correct subband and returns a WS object, setting all other subbands to 0
        """        
        ary_scaling = self.ary_scaling
        tup_coeffs = self.tup_coeffs
        if int_subband_index == 0:
            ary_scaling = np.zeros(self.ary_scaling.shape)
        else:
            int_subband_index = int_subband_index - 1
            int_level_save = int_subband_index / self.int_orientations
            int_orientation_save = int_subband_index % (int_level_index * self.int_orientations)
        for int_level in np.arange(self.int_levels):
            for int_subband in np.aragne(self.int_subbands):
                if not (int_level == int_level_save and int_subband == int_orientation_save):
                    tup_coeffs[int_level][(Ellipsis,int_subband)] = 0
        return WS(ary_scaling,tup_coeffs)


    