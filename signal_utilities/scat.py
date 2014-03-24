#!/usr/bin/python -tt
import numpy as np
from operator import add
from numpy import concatenate as cat

from py_utils.btree import BTree

from py_utils.signal_utilities.sig_utils import downsample_slices
import itertools as it

class Scat(object):
    """
    Scattering class for storing and retreiving and performing operations on scattering 
    coefficients.
    """
    def __init__(self,root_node,int_orientations,int_levels,depth):
        """
        Class constructor for Scattering object,
        coeffs_index is a dictionary of path lists used to look-up
        and index in the coeff_tree
        """
        self.root_node = root_node
        self.int_orientations = int_orientations
        self.max_transform_levels = int_levels
        self.depth = depth
        
    def retrieve(self,subband_paths):   
        """Given a list of lists (subband_paths), compute the indices
        and return a list of scattering coefficients
        """
        return [self.coeffs_tree[self.coeffs_index[subband_path]] for 
                subband_path in subband_paths]

