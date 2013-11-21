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

    def lev_ori_from_subband(self,int_subband_index):
        """
        Given the subband index (>0), compute the level and orientation indices. Note, 0 
        corresponds to scaling function.
        """        
        if int_subband_index == 0:
            raise Exception("index 0 corresponds to scaling function!")    
        else:
            int_level = int_subband_index / self.int_orientations
            int_orientation = int_subband_index % (int_level * self.int_orientations)
        return int_level, int_orientation
    
    def suppress_other_subbands(self,int_subband_index):
        """
        For a given subband index, fetches the correct subband and returns a WS object, setting 
        all other subbands to 0
        """        
        ary_scaling = self.ary_scaling
        tup_coeffs = self.tup_coeffs
        if int_subband_index == 0:
            ary_scaling = np.zeros(self.ary_scaling.shape)
        else:
            int_subband_index = int_subband_index - 1
            int_level_s, int_orientation_s = lev_ori_from_subband(int_subband_index)
        for int_level in np.arange(self.int_levels):
            for int_subband in np.aragne(self.int_subbands):
                if not (int_level == int_level_s and int_subband == int_orientation_s):
                    tup_coeffs[int_level][(Ellipsis,int_subband)] = 0
        return WS(ary_scaling,tup_coeffs)

    def get_subband(self,int_subband_index):
        """
        For a given subband index, returns the corresponding subband as ndarray
        """        
        int_level, int_orientation = lev_ori_from_subband(int_subband_index)
        return self.tup_coeffs[int_level][(Ellipsis,int_subband)]
    
    def set_subband(self,int_subband_index,value):    
        int_level, int_orientation = lev_ori_from_subband(int_subband_index)
        self.tup_coeffs[int_level][(Ellipsis,int_subband)] = value
        
