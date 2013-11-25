#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy
from numpy import mean
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
            self.y = dict_in['y'].flatten()
            self.x = dict_in['x'].flatten()
        x_n = dict_in['x_n'].flatten()
        if y.shape != x.shape or x_n.shape != x.shape:
            raise Exception ("unequal array sizes")
        else:
            value = mean(((x_n - x)**2) / y)
        self.data.append(value)
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return NMISE(ps_parameters,str_section)
