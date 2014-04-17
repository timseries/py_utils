#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy.linalg import norm
import fmetrics as fm
from numpy import conj, arange
from numpy.fft import fftn

from py_utils.signal_utilities.sig_utils import crop_center
from py_utils.section_factory import SectionFactory as sf

class FourierCorrelation(Metric):
    """
    Computes the fourier ring/shell correlation
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(FourierCorrelation,self).__init__(ps_parameters,str_section)        
        self.x_f = None #ground truth dft
        self.fmetrics = sf.create_section(ps_parameters, self.get_val('fmetrics',False))
        
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if self.data == []:
            self.fmetrics.compute_support(dict_in)
            #must use fortran ordering, since this is what matlab uses
            #and we've computed the fourier shell indices assuming this.
            self.x_f = np.ravel(self.fmetrics.x_f,order='F')
            self.x_f_shape=self.fmetrics.x_f.shape
            self.fmetrics.compute_support(dict_in)
        x_n_f = dict_in['x_n']
        if x_n_f.shape != self.x_f.shape:
            x_n_f = crop_center(x_n_f,self.x_f_shape)
        x_n_f = np.ravel(fftn(x_n_f),order='F')
        value = tuple(np.real([np.vdot(np.take(x_n_f,self.fmetrics.s_indices[k]),
                        np.take(self.x_f,self.fmetrics.s_indices[k])) / \
                 norm(np.take(self.x_f,self.fmetrics.s_indices[k]).flatten(),2) / \
                 norm(np.take(x_n_f,self.fmetrics.s_indices[k]).flatten(),2) \
                 for k in xrange(self.fmetrics.K)]))
        self.data.append(tuple(value))
        super(FourierCorrelation,self).update()

    def save(self,str_path='/home/fcorrel/'):
        self.fmetrics.save_bin_csv(self.data,str_path)
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return FourierCorrelation(ps_parameters,str_section)
