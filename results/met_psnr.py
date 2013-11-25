#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy import max as nmax, log10
class PSNR(Metric):
    """
    PSNR metric class, for storing a single number vs iteration.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(PSNR,self).__init__(ps_parameters,str_section)        
        self.x = None #ground truth
        self.peak = self.get_val('peak',True)

    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if self.data == []:
            self.x = dict_in['x'].flatten()
            if self.peak == 0:
                self.peak = nmax(self.x)
        x_n = dict_in['x_n'].flatten()
        if x_n.shape != self.x.shape:
            raise Exception ("unequal array sizes")
        else:
            mse = mean((x_n - self.x)**2)
            if mse == 0:
                snr_db = np.inf
            else:    
                snr_db = 10 * log10((self.peak**2)/mse)
            value = mse, snr_db
            self.data.append(value)
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return PSNR(ps_parameters,str_section)
