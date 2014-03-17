#!/usr/bin/python -tt
import numpy as np
from operator import add
from numpy import concatenate as cat

from py_utils.signal_utilities.sig_utils import downsample_slices
import itertools as it

class Scat(object):
    """
    Scattering class for storing and retreiving and performing operations on scattering 
    coefficients.
    """
    def __init__(self,ary_lowpass):
        """
        Class constructor for Scattering object
        """
        self.coeffs_dict={}
        self.coeffs_dict[0]=ary_lowpass
        
    def store(self,ary_lowpass,depth,parent_subband):
        """
        Store the coefficients in the dictionary, and compute
        the path string based on the depth, the parent_subband,
        and the coefficients already present.
        """
        self.coeffs_dict[]