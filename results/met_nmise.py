#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy
from numpy import mean
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center

class NMISE(Metric):
    """
    NMISE metric class, for storing a single number vs iteration.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(NMISE,self).__init__(ps_parameters,str_section)        
        self.y = None #observation
        self.x = None #ground truth

    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if self.data == []:
            if dict_in['y'].shape != dict_in['x'].shape:
                self.x = crop_center(dict_in['x'],dict_in['y'].shape).flatten()
                self.y = dict_in['y'].flatten()
            else:
                self.x = dict_in['x'].flatten()
                self.y = dict_in['y'].flatten()
        if dict_in['y'].shape != dict_in['x_n'].shape:                
            x_n = crop_center(dict_in['x_n'],dict_in['y'].shape).flatten()
        else:
            x_n = dict_in['x_n'].flatten()
        value = mean(((x_n - self.x)**2) / self.y)
        self.data.append(value)
        super(NMISE,self).update()    
    class Factory:
        def create(self,ps_parameters,str_section):
            return NMISE(ps_parameters,str_section)
