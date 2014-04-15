#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy import sum as nsum, arange
import fmetrics as fm
from numpy import conj
from numpy.fft import fftn
from py_utils.section_factory import SectionFactory as sf

class RER(Metric):
    """
    RER metric class.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(RER,self).__init__(ps_parameters,str_section)        
        self.x_f = None #ground truth dft
        self.fmetrics = sf.create_section(ps_parameters, self.get_val('fmetrics',False))
        
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if self.data == []:
            self.fmetrics.compute_support(dict_in)
            self.x_f = self.fmetrics.x_f.flatten()
            self.fmetrics.compute_support(dict_in)
        x_n_f = fftn(dict_in['x_n']).flatten()
        if x_n_f.shape != self.x_f.shape:
            raise Exception ("unequal array sizes")
        else:
            d_e_bar = self.x_f - x_n_f
            d_e_bar = conj(d_e_bar) * d_e_bar
            e_bar = conj(self.x_f) * self.x_f
            G = [nsum(np.take(e_bar,self.fmetrics.s_indices[k])) for k in arange(self.fmetrics.K)]
            value = tuple([(G[k] - nsum(np.take(d_e_bar,self.fmetrics.s_indices[k])))/G[k] \
                           for k in arange(self.fmetrics.K)])
            self.data.append(value)
            super(RER,self).update()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return RER(ps_parameters,str_section)
