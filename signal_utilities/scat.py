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
    def __init__(self,node,int_orientations,int_levels,depth):
        """
        Class constructor for Scattering object,
        coeffs_index is a dictionary of path lists used to look-up
        and index in the coeff_tree
        """
        if depth > int_levels:
            ValueError('depth of scattering transform greater than number of levels')
        self.coeffs_tree=[]
        self.coeffs_index={}
        self.num_scats = 0
        self.int_orientations = int_orientations
        self.int_levels = int_levels #tranform levels
        self.m1_subbands = self.int_orientations * self.int_levels
        self.depth = depth
        self.bobs = [np.array([self.int_levels-(m-1)-i for i in xrange(self.int_levels-(m-1))])
                             for m in xrange(self.depth)]#bracket offetting boundaries, at each level
        self.csbob = [np.cumsum(bob) for bob in self.bobs]
        self.path_boundaries = get_path_boundaries()
        self.coeffs_tree[self.num_scats]=ary_lowpass
        self.coeffs_index[[self.num_scats]]=self.num_scats

        #tree-based attempt
        self.root = node
        
    def store(self):
        """
        Store the coefficients in the dictionary, and compute
        the path string based on the depth, the parent_subband,
        and the coefficients already present.
        """
        self.num_scats += 1
        self.coeffs_tree[self.num_scats]=ary_lowpass
        self.coeffs_index[self.get_subband_path(level)] = self.num_scats

    def retrieve(self,subband_paths):   
        """Given a list of lists (subband_paths), compute the indices
        and return a list of scattering coefficients
        """
        return [self.coeffs_tree[self.coeffs_index[subband_path]] for 
                subband_path in subband_paths]

