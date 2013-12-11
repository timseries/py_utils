#!/usr/bin/python -tt
from numpy import log10
from numpy.linalg import norm
from py_utils.results.metric import Metric
class ISNR(Metric):
    """
    Base class for defining a metric
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(ISNR,self).__init__(ps_parameters,str_section)
        self.y = None #observation
        self.x = None #ground truth
        
    def update(self,dict_in):
        if self.data == []:
            self.y = dict_in['y'].flatten()
            self.x = dict_in['x'].flatten()
        x_n = dict_in['x_n'].flatten()
        value = 10 * log10((norm(self.y - self.x,2)**2)/(norm(x_n - self.x,2)**2))
        self.data.append(value)
        super(ISNR,self).update()
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return ISNR(ps_parameters,str_section)
