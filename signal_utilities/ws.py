#!/usr/bin/python -tt
import numpy as np
from copy import deepcopy
class WS(object):
    """
    WS class for storing and retreiving and performing operations on wavelet subbands. 
    """
    
    def __init__(self,ary_scaling,tup_coeffs):
        """
        Class constructor for DTCWT
        """
        self.ary_scaling = ary_scaling.copy()
        self.tup_coeffs = deepcopy(tup_coeffs)
        self.ary_size = np.dot(2,ary_scaling.shape) 
        self.int_levels = len(tup_coeffs)
        self.int_dimension = ary_scaling.ndim
        self.int_orientations = tup_coeffs[0].shape[-1]
        self.int_subbands = self.int_levels * self.int_orientations + 1

    def lev_ori_from_subband(self,int_subband_index):
        """
        Given the subband index (>=0), compute the level and orientati
        !23Gamma
        on indices. Note, 0 
        corresponds to scaling function.
        """        
        if int_subband_index == 0:
            raise Exception("index 0 corresponds to scaling function!")    
        else:
            int_subband_index -= 1
            int_level = int_subband_index / self.int_orientations
            int_orientation = int_subband_index % self.int_orientations
        return int_level, int_orientation
    
    def one_subband(self,int_subband_index):
        """
        Returns a ws object which is a copy of this one, except all of the subbands except
        int_subband_index have been set to 0.
        """ 
        #create new object
        ws_one_subband = WS(self.ary_scaling,self.tup_coeffs)
        if int_subband_index != 0:
            ws_one_subband.ary_scaling = np.zeros(self.ary_scaling.shape)
            int_level_s, int_orientation_s = self.lev_ori_from_subband(int_subband_index)
        for int_level in np.arange(self.int_levels):
            for int_orientation in np.arange(self.int_orientations):
                if int_subband_index == 0 or \
                  (not (int_level == int_level_s and int_orientation == int_orientation_s)):
                    ws_one_subband.tup_coeffs[int_level][(Ellipsis,int_orientation)] = 0
        return ws_one_subband

    def get_subband(self,int_subband_index):
        """
        For a given subband index, returns the corresponding subband as ndarray
        """ 
        if int_subband_index == 0:
            return self.ary_scaling
        else:
            int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
        return self.tup_coeffs[int_level][(Ellipsis,int_orientation)]
    
    def set_subband(self,int_subband_index,value):    
        if int_subband_index == 0:
            self.ary_scaling = value
        else:
            int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
            self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = value
        
