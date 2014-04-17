#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy import sum as nsum, arange
import fmetrics as fm
from numpy import conj
from numpy.fft import fftn

from py_utils.signal_utilities.sig_utils import crop_center
from py_utils.section_factory import SectionFactory as sf

class RER(Metric):
    """
    RER (relative energy regain) metric class.
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
            self.x_f = np.ravel(self.fmetrics.x_f,order='F')
            self.x_f_shape=self.fmetrics.x_f.shape
            self.fmetrics.compute_support(dict_in)
        x_n_f = dict_in['x_n']
        if x_n_f.shape != self.x_f.shape:
            x_n_f = crop_center(x_n_f,self.x_f_shape)
        x_n_f = np.ravel(fftn(x_n_f),order='F')    
        d_e_bar = self.x_f - x_n_f
        d_e_bar = conj(d_e_bar) * d_e_bar
        e_bar = conj(self.x_f) * self.x_f
        G = [nsum(np.take(e_bar,self.fmetrics.s_indices[k])) for k in xrange(self.fmetrics.K)]
        value = tuple(np.real([(G[k] - nsum(np.take(d_e_bar,self.fmetrics.s_indices[k])))/G[k] \
                       for k in arange(self.fmetrics.K)]))
        self.data.append(value)
        super(RER,self).update()

    def save(self,str_path='/home/fcorrel/'):
        self.fmetrics.save_bin_csv(self.data,str_path)
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return RER(ps_parameters,str_section)
