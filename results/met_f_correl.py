#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy.linalg import norm
import fmetrics as fm
from numpy import conj, arange
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
            self.x_f = dict_in['x_f'].flatten()
            self.fmetrics.compute_support(dict_in)
        x_n_f = dict_in['x_n_f'].flatten()
        if x_n_f.shape != self.x_f.shape:
            raise Exception ("unequal array sizes")
        else:
            value = [np.dot(np.take(self.x_f,self.fmetrics.s_indices[k]).flatten(), \
                              np.take(x_n_f,self.fmetrics.s_indices[k]).flatten()) / \
                     norm(np.take(self.x_f,self.fmetrics.s_indices[k]).flatten(),2) / \
                     norm(np.take(x_n_f,self.fmetrics.s_indices[k]).flatten(),2) \
                     for k in arange(self.fmetrics.K)]
            self.data.append(tuple(value))
            super(FourierCorrelation,self).update()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return FourierCorrelation(ps_parameters,str_section)
