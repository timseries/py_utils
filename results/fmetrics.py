#!/usr/bin/python -tt
from py_utils.section import Section
from py_utils.results.metric import Metric
import numpy as np
from numpy import arange, floor, meshgrid as mg, asarray, sqrt, flatnonzero as nz
from numpy.linalg import norm
from numpy.fft import ifftshift, fftn
import csv

from py_utils.results.defaults import DEFAULT_CSV_EXT

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
            #floating point version, for computing floating point fractions below
            self.Kf=np.asfarray(self.K) 
        self.epsilon = self.get_val('epsilon',True)
        if self.epsilon == 0:
            self.epsilon = 1e-6
        self.s_indices = None #a list of 1D arrays
        self.weights = None #corresponding list number of elements
        
    def save_bin_csv(self,fourier_data,str_output_path):
        '''save bins across columns, iterations down rows, assuming 
        fourier_dtaa is a length-n list of bin-tuples, in a csv file
        
        '''
        int_rows = len(fourier_data)
        bins=len(fourier_data[0])
        bins_f=np.asfarray(bins)
        headers = [col/bins_f for col in xrange(bins)] 
        headers.insert(0,'n')
        table = [[j] + [bin_ for bin_ in fourier_data[j]] 
                 for j in xrange(int_rows)]
        with open(str_output_path+'.'+DEFAULT_CSV_EXT, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(table)


    def compute_support(self, dict_in):
        self.x_f = fftn(dict_in['x'])
        int_dims = self.x_f.ndim
        ary_shape = self.x_f.shape
        gridpts = mg(*[ifftshift(arange(int(np.floor(-(ary_shape[d]-1)/2.0)),
                                        int(np.floor((ary_shape[d]-1)/2.0))+1)) \
              for d in arange(int_dims)])
        radius=0.0      
        # print 'gripts info'
        # print gridpts

        for d in arange(int_dims):
            radius += (2.0 * gridpts[d] / ary_shape[d])**2.0
        radius = sqrt(radius)
        # print 'radius info'
        # print radius

        self.s_indices = [nz((k/self.Kf-self.epsilon < radius) * (radius <= (k+1)/self.Kf)) for k in xrange(self.K)]
        self.weights = [len(self.s_indices[k]) for k in xrange(self.K)]
                                  
    class Factory:
        def create(self,ps_parameters,str_section):
            return FMetrics(ps_parameters,str_section)