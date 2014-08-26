#!/usr/bin/python -tt
import numpy as np
from operator import add
from numpy import concatenate as cat
from copy import deepcopy
# import theano
# from theano import tensor as T
from py_utils.signal_utilities.sig_utils import downsample_slices
import itertools as it

import pdb

class WS(object):
    """
    WS class for storing and retreiving and performing operations on wavelet subbands. 
    """
    def __init__(self,ary_lowpass,tup_coeffs,tup_scaling=None):
        """
        Class constructor for WS
        """
        self.ary_lowpass = ary_lowpass.copy()
        self.tup_coeffs = deepcopy(tuple(tup_coeffs))
        if tup_scaling:
            self.tup_scaling = deepcopy(tup_scaling)
        self.ary_shape = tuple(2*np.array(tup_coeffs[0][(Ellipsis,0)].shape))
        self.int_levels = len(tup_coeffs)
        self.int_dimension = ary_lowpass.ndim
        self.ndim = ary_lowpass.ndim
        self.ds_slices = downsample_slices(self.int_dimension)
        self.int_orientations = tup_coeffs[0].shape[-1]
        self.int_subbands = self.int_levels * self.int_orientations + 1
        self.dims = None #dimensions of all of the subbands stored in a list
        self.size = None #total number of elements
        self.get_dims()
        self.ws_vector = None

    def lev_ori_from_subband(self,int_subband_index,ori_degrees=False):
        """
        Given the subband index (>=0), compute the level and orientation indices. Note, 0 
        corresponds to lowpass image.
        If ori_degrees==True, then convert the orientation index to a degree value.
        """        
        if int_subband_index == 0:
            raise Exception("index 0 corresponds to lowpass subband")    
        else:
            int_subband_index -= 1
            int_level = int_subband_index / self.int_orientations
            int_orientation = int_subband_index % self.int_orientations
            if ori_degrees:
                if self.int_orientations==6:
                    int_orientation = {
                        0: 15,
                        1: 45,
                        2: 75,
                        3: -75,
                        4: -45,
                        5: -15
                        }.get(int_orientation, 15)
        return int_level, int_orientation
    
    def one_subband(self,int_subband_index):
        """
        Returns a ws object which is a copy of this one, except all of the subbands except
        int_subband_index have been set to 0.
        """ 
        #create new object
        # ws_one_subband =  WS(self.ary_lowpass,self.tup_coeffs)
        ws_one_subband = 0 * self
        ws_one_subband.set_subband(int_subband_index, self.get_subband(int_subband_index))
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

    def get_upsampled_parent(self,s):
        """
        Returns the upsampled parent of s, by copying the elements of the parent.
        The output array should be the same size as s.
        """
        if s>=self.int_subbands-self.int_orientations:
            #we actually need to downsample in this case, as we're using the scaling coefficients
            subband_index = 0
        elif s<0:
            raise ValueError('Invalid subband index: ' + str(s))
        else:
            subband_index = s+self.int_orientations
        if subband_index==0: #downsample by averaging the lowpass suband
            s_parent_us = self.subband_group_sum(subband_index,'children',True,False)[self.ds_slices[0]]
        else:
            s_parent = self.get_subband(subband_index)
            s_parent_us = np.zeros(2*np.asarray(s_parent.shape),dtype=s_parent.dtype)
            for j in xrange(len(self.ds_slices)):
                s_parent_us[self.ds_slices[j]]=s_parent
        return s_parent_us
        
    def subband_group_sum(self,s,group_type,average=True, energy=True):
        """
        Computes the sum (or average) (energy) of one of two group types for subband s
        Group_type can be either 'parent_children' or  'children' or 'parent_child'
        """
        if group_type != 'children':
            w_parent = self.get_upsampled_parent(s) #THIS BAD, we have two functions calling each other (potentially)
            if energy:
                w_parent = np.abs(w_parent)**2
        w_child = self.get_subband(s)   
        if energy:
            w_child = np.abs(w_child)**2
        if group_type == 'parent_children' or group_type == 'children':
            w_child = np.sum(cat([w_child[self.ds_slices[i]][...,np.newaxis] 
                                  for i in xrange(len(self.ds_slices))],
                                  axis=self.int_dimension),axis=self.int_dimension)
            w_child_us = np.zeros(2*np.asarray(w_child.shape))
            for j in xrange(len(self.ds_slices)):
                w_child_us[self.ds_slices[j]] = w_child
            del w_child    
            if group_type == 'parent_children':             
                divisor = 2.0**self.int_dimension + 1
            else: #children only
                w_parent = 0
                divisor = 2.0**self.int_dimension    
        elif group_type == 'parent_child':
            w_child_us = w_child
            divisor = 2.0
        w_parent += w_child_us
        if average:
            w_parent /= divisor
        return w_parent    

    def modulus(self):
        """Takes the modulus across all of the subands, and returns a new WS object
        """
        return WS(np.abs(self.ary_lowpass),np.abs(self.tup_coeffs))
    
    def energy(self):
        """Takes the modulus across all of the subands, and returns a new WS object
        """
        return WS(np.abs(self.ary_lowpass)**2,tuple([np.abs(ary_coeffs)**2 for ary_coeffs in self.tup_coeffs]))

    def cast(self,str_dtype):
        """Returns a copy of this object with all values cast to dtype
        """
        return WS(np.array(self.ary_lowpass,dtype=str_dtype),
                  tuple([np.array(ary_coeffs,dtype=str_dtype) 
                         for ary_coeffs in self.tup_coeffs]))
        
    def __add__(self,summand):
        if summand.__class__.__name__=='WS':
            return WS(self.ary_lowpass+summand.ary_lowpass,
                      tuple([self.tup_coeffs[j] + summand.tup_coeffs[j] for j in xrange(self.int_levels)]))
        elif (summand.__class__.__name__=='ndarray' and 
              summand.shape[-1] == 1 and 
              summand.shape[0] == self.int_subbands):
            ws_temp = self*1 #temporary copy
            for s in xrange(self.int_subbands-1,-1,-1):
                ws_temp.set_subband(s,ws_temp.get_subband(s)+summand[s])
            return ws_temp    
        else:    
            return WS(self.ary_lowpass+summand,tuple([ary_coeffs+summand for ary_coeffs in self.tup_coeffs]))

    def __radd__(self,summand):
        return self.__add__(summand)    

    def __rsub__(self,subtrahend):
        return self.__sub__(subtrahend)    

    def __rmul__(self,multiplicand):
        return self.__mul__(multiplicand)    

    def __sub__(self,subtrahend):
        return self.__add__(-1.0*subtrahend)    


    def nonzero(self):
        """returns a WS object with True in all of the non-zero positions
        """
        return self.cast('bool')

    def __invert__(self):
        return 1-self.cast('bool')
    
    def __mul__(self,multiplicand):
        if multiplicand.__class__.__name__=='WS':
            return WS(multiplicand.ary_lowpass*self.ary_lowpass,
                      tuple([self.tup_coeffs[j] * multiplicand.tup_coeffs[j] 
                             for j in xrange(self.int_levels)]))
        else:
            return WS(multiplicand*self.ary_lowpass, tuple(multiplicand*np.array(self.tup_coeffs)))

    def __div__(self,divisor):
        if divisor.__class__.__name__=='WS':
            return WS(self.ary_lowpass / divisor.ary_lowpass,
                      tuple([self.tup_coeffs[j] / divisor.tup_coeffs[j] 
                             for j in xrange(self.int_levels)]))
        else:
            return WS(self.ary_lowpass / divisor, tuple(np.array(self.tup_coeffs) / divisor))
        
    def invert(self):
        return WS(1.0/self.ary_lowpass,tuple(1.0/np.array(self.tup_coeffs)))
        
    def precision(self,regularizer=.01):
        """Generate a precision WS object from this WS object
        """
        return WS(1.0/(np.abs(self.ary_lowpass)**2+regularizer),
                  1.0/(np.abs(self.tup_coeffs)**2+regularizer))
        
    def set_subband(self,int_subband_index,value):
        if int_subband_index == 0:
            self.ary_lowpass = value
        else:
            int_level, int_orientation = self.lev_ori_from_subband(int_subband_index)
            self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = value

    # def f_flatten(self):
    #     """
    #     Returns the wavelet object as a vector (Nx1 ndarray)
    #     """
    #     if self.dims == None:
    #         self.dims = [ary_lowpass.shape]
    #         self.dims = self.dims.append([self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape \
    #                      for int_level,int_orientation in zip(self.int_levels,self.int_orientations)])
    #     products = map(product, self.dims)
    #     # flattening:
    #     xs = [T.tensor3()]
    #     for _ in self.dims:
    #         xs.append(T.tensor3())
    #         combined = T.concatenate([T.flatten(x) for x in xs])
    #     flatten = theano.function(xs, combined)
    #     return flatten
            
    # def f_unflatten(self):
    #     '''
    #     Returns the original array, stores in the ary_lowpass and tup_coeffs objects. 
    #     Assumes flattens is called first.
    #     '''
    #     products = map(self.product, dims)
    #     # now inverse mapping:
    #     inverse_input = T.vector()
    #     inverse_output = []
    #     accum = 0
    #     for prod, dim in zip(products, self.dims) :
    #         inverse_output.append(T.reshape(inverse_input[accum:accum+prod], dim))
    #         accum += prod
    #     unflatten = theano.function([inverse_input], inverse_output)
    #     return unflatten
        
    def product(self,ds):
        out = 1
        for d in ds:
            out *= d
        return out

    def flatten(self,lgc_real=False):
        '''
        Flattens the wavelet object as a vector (Nx1 ndarray) as a member (a vector view of the data). 
        lgc_real: whether or not to use purely real ouput. In this case, real/imag parts are stored consecutively.
        If duplicate=True, then we simply copy the elements twice. This is useful for thresholding using the complex modulus.
        '''
        int_wav_cplx = self.is_wavelet_complex()
        int_low_cplx = self.is_lowpass_complex()
        #int_if is either 1 or 2, the increment factor variable.
        #int_if is the 1 by default. If a real output is needed (lgc_real) from a complex input, then we need to store
        #the real an imaginary part in alternate locations in the vector. 
        int_if_low = int_low_cplx * lgc_real + 1
        int_if_wav = int_wav_cplx * lgc_real + 1
        int_len_ws_vector = self.size*(int_if_wav)-(int_if_low==1)*(int_if_wav==2)*np.prod(self.ary_lowpass.size)
        ws_vector_dtype='complex64'
        if lgc_real or not (int_low_cplx or int_wav_cplx):
            ws_vector_dtype='float32'
        if self.ws_vector == None or int_len_ws_vector!=self.ws_vector.size:
            self.ws_vector = np.zeros(int_len_ws_vector,dtype=ws_vector_dtype)
        int_stride = self.ary_lowpass.size*int_if_low
        int_p_stride = 0
        #the lowpass image
        if int_if_low==2:
            self.ws_vector[int_p_stride:int_stride:2] = np.real(self.ary_lowpass.flatten())
            self.ws_vector[int_p_stride+1:int_stride+1:2] = np.imag(self.ary_lowpass.flatten())
        else:
            self.ws_vector[int_p_stride:int_stride:1] = self.ary_lowpass.flatten()
        #the highpass coefficients
        for int_level,int_orientation in self.get_levs_ors():
            sz = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].size
            int_p_stride = int_stride
            int_stride = sz*int_if_wav + int_p_stride
            ary_tup_coeffs = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].flatten()
            if int_if_wav==2:
                self.ws_vector[int_p_stride:int_stride:2] = np.real(ary_tup_coeffs)
                self.ws_vector[int_p_stride+1:int_stride+1:2] = np.imag(ary_tup_coeffs)
            else:    
                self.ws_vector[int_p_stride:int_stride:1] = ary_tup_coeffs
        return self.ws_vector

    def unflatten(self,new_ws_vector=None,lgc_real=False):
        '''
        Unflattens the new_ws_vector back in this ws object, if provided. Otherwise
        unflatten the self.ws_vector attribute.
        '''
        if new_ws_vector!=None:
            self.ws_vector=new_ws_vector
        int_wav_cplx = self.is_wavelet_complex()
        int_low_cplx = self.is_lowpass_complex()
        int_if_low = int_low_cplx * lgc_real + 1
        int_if_wav = int_wav_cplx * lgc_real + 1
        int_p_stride = 0
        int_stride = self.ary_lowpass.size*int_if_low
        if int_if_low==2:
            self.ary_lowpass = (self.ws_vector[int_p_stride:int_stride:2] + 
                                1.0j*self.ws_vector[int_p_stride+1:int_stride+1:2]).reshape(self.ary_lowpass.shape)
        else:
            self.ary_lowpass = self.ws_vector[int_p_stride:int_stride:1].reshape(self.ary_lowpass.shape)
        for int_level,int_orientation in self.get_levs_ors():
            dim = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape
            sz = self.tup_coeffs[int_level][(Ellipsis,int_orientation)].size
            int_p_stride = int_stride
            int_stride = sz*int_if_wav + int_p_stride
            if int_if_wav==2:
                self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = \
                  (self.ws_vector[int_p_stride:int_stride:2].reshape(dim) + \
                   1.0j*self.ws_vector[int_p_stride+1:int_stride+1:2].reshape(dim))
            else:
                self.tup_coeffs[int_level][(Ellipsis,int_orientation)] = \
                self.ws_vector[int_p_stride:int_stride:1].reshape(dim)
        return self        
                  
    def get_dims(self):
        '''Store the dimensions of the subbands as a list, (self.dims)
         as well as the total number of coefficients (self.size)
        '''
        if self.dims == None:
            self.dims = [self.ary_lowpass.shape]
            self.dims = self.dims + [self.tup_coeffs[int_level][(Ellipsis,int_orientation)].shape \
                                     for int_level,int_orientation in self.get_levs_ors()]
            self.size = sum([np.prod(self.dims[i]) for i in np.arange(len(self.dims))])

    def downsample_scaling(self):        
        """Return the spacial averaged version of the scaling indices 
        Downsampled by a factor of 2 in each dimension.
        """    
        averaging_factor = 2.0**self.int_dimension
        ls_downsample_scaling=[]
        for level in xrange(self.int_levels):
            scaling_factor = (2.0*np.sqrt(2))**(level+1)
            ds_scaling_coeffs = np.zeros(np.array(self.tup_scaling[level].shape)/2)
            for ds_slice in self.ds_slices:
                ds_scaling_coeffs+=self.tup_scaling[level][ds_slice]
            ls_downsample_scaling.append(ds_scaling_coeffs/(averaging_factor*scaling_factor))
        return tuple(ls_downsample_scaling)

    def get_levels(self):
        return self.int_levels

    def get_levs_ors(self):
        return it.product(np.arange(self.int_levels),np.arange(self.int_orientations))

    def is_wavelet_complex(self):
        return np.iscomplexobj(self.tup_coeffs[0])

    def is_lowpass_complex(self):
        return np.iscomplexobj(self.ary_lowpass)