#!/usr/bin/python -tt
import numpy as np
from copy import deepcopy
import theano
from theano import tensor as T
import itertools as it

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
        self.dims = None #dimensions of all of the subbands
        self.N = None #total number of elements
        self.ws_vector = None

    def lev_ori_from_subband(self,int_subband_index):
        """
        Given the subband index (>=0), compute the level and orientation indices. Note, 0 
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

    def f_flatten(self):
        '''
        Returns the wavelet object as a vector (Nx1 ndarray)
        '''
        if self.dims == None:
            self.dims = [ary_scaling.shape]
            self.dims = self.dims.append([self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape \
                         for int_level,int_orientation in zip(self.int_levels,self.int_orientations)])
        products = map(product, self.dims)
        # flattening:
        xs = [T.tensor3()]
        for _ in self.dims:
            xs.append(T.tensor3())
            combined = T.concatenate([T.flatten(x) for x in xs])
        flatten = theano.function(xs, combined)
        return flatten
            
    def f_unflatten(self):
        '''
        Returns the original array, stores in the ary_scaling and tup_coeffs objects. 
        Assumes flattens is called first.
        '''
        products = map(self.product, dims)
        # now inverse mapping:
        inverse_input = T.vector()
        inverse_output = []
        accum = 0
        for prod, dim in zip(products, self.dims) :
            inverse_output.append(T.reshape(inverse_input[accum:accum+prod], dim))
            accum += prod
        unflatten = theano.function([inverse_input], inverse_output)
        return unflatten
        
    def product(self,ds):
        out = 1
        for d in ds:
            out *= d
        return out

    def flatten(self,thresh=False):
        '''
        Flattens the wavelet object as a vector (Nx1 ndarray) as a member (a vector view of the data)
        '''
        self.get_dims()
        #allocate the vector interface once
        if self.ws_vector == None:
            self.is_complex = 0
            if str(self.tup_coeffs[0][(Ellipsis,0)].dtype)[0:7]=='complex':
                self.is_complex = 1
            int_len_ws_vector = self.N*(self.is_complex+1)
            self.ws_vector = np.zeros(int_len_ws_vector,dtype='float32')
        int_this_stride = np.product(self.ary_scaling.shape)*(self.is_complex+1)
        int_last_stride = 0
        self.ws_vector[int_last_stride:int_this_stride:2] = np.real(self.ary_scaling.flatten())
        if thresh:
            self.ws_vector[int_last_stride+1:int_this_stride+1:2] = np.real(self.ary_scaling.flatten())
        else:
            self.ws_vector[int_last_stride+1:int_this_stride+1:2] = np.imag(self.ary_scaling.flatten())
        for int_level,int_orientation in self.get_levs_ors():
            dim = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape
            int_last_stride = int_this_stride
            int_this_stride = np.product(dim) + int_last_stride
            self.ws_vector[int_last_stride:int_this_stride] = \
              self.tup_coeffs[int_level][(Ellipsis,int_orientation)].flatten()

    def unflatten(self):
        '''
        Stores the ws_vector back in the ws object, assumes w_ve
        '''
        self.get_dims()
        int_this_stride = np.product(self.ary_scaling.shape)*(self.is_complex+1)
        int_last_stride = 0
        self.ary_scaling = self.ws_vector[int_last_stride:int_this_stride:2].reshape(self.ary_scaling.shape)
        for int_level,int_orientation in self.get_levs_ors():
            dim = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape
            int_last_stride = int_this_stride
            int_this_stride = np.product(dim)*(self.is_complex+1) + int_last_stride
            self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = \
              self.ws_vector[int_last_stride:int_this_stride:2].reshape(dim) + \
              1j*self.ws_vector[int_last_stride+1:int_this_stride+1:2].reshape(dim)

    def get_dims(self):
        if self.dims == None:
            self.dims = [self.ary_scaling.shape]
            self.dims = self.dims + [self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape \
                                     for int_level,int_orientation in self.get_levs_ors()]
            self.N = sum([np.prod(self.dims[i]) for i in np.arange(len(self.dims))])

    def get_levs_ors(self):
        return it.product(np.arange(self.int_levels),np.arange(self.int_orientations))
