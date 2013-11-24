#!/usr/bin/python -tt
from numpy import log10
from numpy.linalg import norm
from numpy.ndarray import flatten
class Parameter(Section):
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
         if self.data == []
             self.y = flatten(dict_in['y'])
             self.x = flatten(dict_in['x'])
         value = 10*log10((norm(y-x,2)^2)/(norm(x_n-x,2)^2))
         self.data.append(value)
         