#!/usr/bin/python -tt
from py_utils.results.metric import Metric

class Scalar(Metric):
    """
    Scalar metric class, for storing a single number or label vs iteration or sample index.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(Scalar,self).__init__(ps_parameters,str_section)        
        self.lgc_stop = False
        
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector or list and stop.
        """
        if not self.lgc_stop:
            value = dict_in[self.get_val('key',False)]
            if (value.__class__.__name__ =='ndarray' or 
                value.__class__.__name__ =='list'):
                if len(value)>1:
                    self.data = value
                    self.lgc_stop = True
            else:                
                self.data.append(value)
            super(Scalar,self).update()    
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return Scalar(ps_parameters,str_section)
