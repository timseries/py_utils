#!/usr/bin/python -tt
from py_utils.results.metric import Metric

class Scalar(Metric):
    """
    Scalar metric class, for storing a single number vs iteration.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(Scalar,self).__init__(ps_parameters,str_section)        
        self.lgc_stop = False
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector and stop.
        """
        if not self.lgc_stop:
            value = dict_in[self.get_val('key',False)]
            if value.shape[0] > 1:
                self.data = value
                self.lgc_stop = True
            else:                
                self.data.append(value)
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return Scalar(ps_parameters,str_section)
