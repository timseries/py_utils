#!/usr/bin/python -tt
from numpy.linalg import norm
from py_utils.results.metric import Metric

import pdb

class L2E(Metric):
    """L2E metric defined in Holland et al "Reducing Acquisition Times..."
    
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for L2E.
        """       
        super(L2E,self).__init__(ps_params,str_section)
        
    def update(self,dict_in):
        """Update this metric.

        Args: 
        dict_in (dict): Must have the fields:
            'theta': Ground truth phase image
            'theta_n': Current iterate of phase image.
            'mask' (ndarray, optional): The binary mask
            '
        """
        
        if self.data == []:
            self.mask = 1
            if dict_in.has_key('mask'):
                self.mask = dict_in['mask']
            self.theta = dict_in['theta']
            self.theta_masked_energy = norm((self.mask * dict_in['theta']).flatten(), 2)
        value = norm((self.mask*(dict_in['theta_n'] - self.theta)).flatten(), 2) / self.theta_masked_energy
        super(L2E,self).update(value)
        
    class Factory:
        def create(self,ps_params,str_section):
            return L2E(ps_params,str_section)
