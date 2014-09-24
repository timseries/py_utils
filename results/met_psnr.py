#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy import max as nmax, log10, mean
from py_utils.signal_utilities.sig_utils import crop_center
import pdb

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
            self.xshape = dict_in['x'].shape
            self.x = dict_in['x'].flatten()
            if self.peak == 0:
                self.peak = nmax(self.x)
        if dict_in['x_n'].shape != self.xshape:
            x_n = crop_center(dict_in['x_n'],self.xshape).flatten()
        else:
            x_n = dict_in['x_n'].flatten()
        mse = mean((x_n - self.x)**2)
        if mse == 0:
            snr_db = np.inf
        else:    
            snr_db = 10 * log10((self.peak**2)/mse)
        value = snr_db
        self.data.append(value)
        super(PSNR,self).update()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return PSNR(ps_parameters,str_section)
