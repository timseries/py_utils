#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy import sum as nsum
import fmetrics as fm
from numpy import conj
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
        self.fmetrics = fm.FMetrics(ps_parameters,str_section)
        
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if self.data == []:
            self.x_f = dict_in['x_f'].flatten()
            self.fmetric.compute_support(dict_in)
        x_n_f = dict_in['x_n_f'].flatten()
        if x_n_f.shape != self.x.shape:
            raise Exception ("unequal array sizes")
        else:
            d_e_bar = self.x_f - x_n_f
            d_e_bar = conj(d_e_bar) * d_e_bar
            e_bar = conj(self.x_f) * self.x_f
            Gk = [nsum(e_bar[self.fmetrics.s_sindices[k]]) for k in self.fmetrics.K]
            value = tuple([(G[k] - nsum(d_e_bar[self.fmetrics.s_sindices[k]]))/G[k]\
                           for k in self.fmetrics.K])
            self.data.append(value)
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return RER(ps_parameters,str_section)
