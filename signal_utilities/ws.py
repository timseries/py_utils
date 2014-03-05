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
    
    def __init__(self,ary_lowpass,tup_coeffs,tup_scaling=None):
        """
        Class constructor for DTCWT
        """
        self.ary_lowpass = ary_lowpass.copy()
        self.tup_coeffs = deepcopy(tup_coeffs)
        if tup_scaling:
            self.tup_scaling = deepcopy(tup_scaling)
        self.ary_size = np.dot(2,ary_lowpass.shape) 
        self.int_levels = len(tup_coeffs)
        self.int_dimension = ary_lowpass.ndim
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
            raise Exception("index 0 corresponds to lowpass subband")    
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
        ws_one_subband = WS(self.ary_lowpass,self.tup_coeffs)
        if int_subband_index != 0:
            ws_one_subband.ary_lowpass = np.zeros(self.ary_lowpass.shape)
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
            return self.ary_lowpass
        else:
            int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
        return self.tup_coeffs[int_level][(Ellipsis,int_orientation)]

    def get_subband_sc(self,int_subband_index):
        """
        For a given subband index, returns the corresponding scaling coefficient as ndarray
        """ 
        int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
        return self.tup_scaling[int_level]
    
    def set_subband(self,int_subband_index,value):
        if int_subband_index == 0:
            self.ary_lowpass = value
        else:
            int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
            self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = value

    def f_flatten(self):
        '''
        Returns the wavelet object as a vector (Nx1 ndarray)
        '''
        if self.dims == None:
            self.dims = [ary_lowpass.shape]
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
        Returns the original array, stores in the ary_lowpass and tup_coeffs objects. 
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

    def flatten(self,lgc_real=True,duplicate=False):
        '''
        Flattens the wavelet object as a vector (Nx1 ndarray) as a member (a vector view of the data). 
        lgc_real: whether or not to use purely real ouput. In this case, real/imag parts are stored consecutively.
        If duplicate=True, then we simply copy the elements twice. This is useful for thresholding using the complex modulus.
        '''
        self.get_dims()
        #allocate the vector interface once
        if self.ws_vector == None:
            self.is_complex = 0
            if str(self.tup_coeffs[0][(Ellipsis,0)].dtype)[0:7]=='complex':
                self.is_complex = True
            self.int_if = self.is_complex * lgc_real + 1
            if duplicate:
                self.int_if = 2
            int_len_ws_vector = self.N*(self.int_if)-(self.is_complex*lgc_real)*np.prod(self.dims[0])#counteract double counting of real lowpass
            self.ws_vector = np.zeros(int_len_ws_vector,dtype='float32')
        int_this_stride = np.product(self.ary_lowpass.shape)
        int_last_stride = 0
        #the lowpass image
        self.ws_vector[int_last_stride:int_this_stride:1] = self.ary_lowpass.flatten()
        #the highpass coefficients
        for int_level,int_orientation in self.get_levs_ors():
            dim = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape
            int_last_stride = int_this_stride
            int_this_stride = np.product(dim)*self.int_if + int_last_stride
            ary_tup_coeffs = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].flatten()
            self.ws_vector[int_last_stride:int_this_stride:self.int_if] = np.real(ary_tup_coeffs)
            if self.int_if==2:
                if duplicate:
                    self.ws_vector[int_last_stride+1:int_this_stride:self.int_if] = np.real(ary_tup_coeffs)
                else:    
                    self.ws_vector[int_last_stride+1:int_this_stride:self.int_if] = np.imag(ary_tup_coeffs)
        return self.ws_vector


    def unflatten(self):
        '''
        Stores the ws_vector back in the ws object, assumes w_ve
        '''
        self.get_dims()
        int_last_stride = 0
        int_this_stride = np.prod(self.ary_lowpass.shape)
        self.ary_lowpass = self.ws_vector[int_last_stride:int_this_stride:1].reshape(self.ary_lowpass.shape)
        for int_level,int_orientation in self.get_levs_ors():
            dim = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape
            int_last_stride = int_this_stride
            int_this_stride = np.product(dim)*(self.int_if) + int_last_stride
            if self.int_if==2:
                self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = \
                  (self.ws_vector[int_last_stride:int_this_stride:2].reshape(dim) + \
                   1.0j*self.ws_vector[int_last_stride+1:int_this_stride+1:2].reshape(dim))
            else:
                self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = \
                  self.ws_vector[int_last_stride:int_this_stride:1].reshape(dim)
                  
    def get_dims(self):
        if self.dims == None:
            self.dims = [self.ary_lowpass.shape]
            self.dims = self.dims + [self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape \
                                     for int_level,int_orientation in self.get_levs_ors()]
            self.N = sum([np.prod(self.dims[i]) for i in np.arange(len(self.dims))])

    def get_levs_ors(self):
        return it.product(np.arange(self.int_levels),np.arange(self.int_orientations))
