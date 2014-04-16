#!/usr/bin/python -tt
from py_utils.section import Section
from py_utils.results.metric import Metric
import numpy as np
from numpy import arange, floor, meshgrid as mg, asarray, sqrt, nonzero as nz
from numpy.linalg import norm
from numpy.fft import ifftshift, fftn

class FMetrics(Section):
    """
    FMetrics class, for computing the support of a Fourier ring/shell.
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(FMetrics,self).__init__(ps_parameters,str_section)        
        self.x_f = None #ground truth dft
        self.K = self.get_val('k',True)
        if self.K == 0:
            self.K = 64 #number of bins
        self.epsilon = self.get_val('epsilon',True)
        if self.epsilon == 0:
            self.epsilon = 1e-5
        self.s_indices = None #a list of 1D arrays
        self.weights = None #corresponding list number of elements
        
    def compute_support(self, dict_in):
        self.x_f = fftn(dict_in['x'])
        int_dims = self.x_f.ndim
        ary_shape = self.x_f.shape
        gridpts = mg(*[ifftshift(arange(int(-(ary_shape[d]-1)/2.0),int((ary_shape[d]-1)/2.0))) \
              for d in arange(int_dims)])
        radius=0      
        # print 'gripts info'
        # print gridpts

        for d in arange(int_dims):
            radius += (2 * gridpts[d] / ary_shape[d])**2
        radius = sqrt(radius)
        # print 'radius info'
        # print radius

        self.s_indices = [nz((k/self.K-self.epsilon < radius) * (radius <= (k+1)/self.K)) \
                                  for k in arange(self.K)]
        self.weights = [len(self.s_indices[k]) for k in arange(self.K)]
                                  
    class Factory:
        def create(self,ps_parameters,str_section):
            return FMetrics(ps_parameters,str_section)