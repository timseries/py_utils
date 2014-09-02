#!/usr/bin/python -tt
import numpy as np
from numpy import log10
from numpy.linalg import norm
from py_utils.results.met_subband_metric import SubbandMetric
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center
from py_utils.section_factory import SectionFactory as sf


class SRE(SubbandMetric):
    """
    The subband relative error. Computes the relative energy error all subbands in a wavelet decomposistion.
    \frac{\|\mathbf{w}_j-\mathbf{w}_j^*\|_2^2}{\|\mathbf{w}_j^*\|_2^2}
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for SRE.
        """       
        super(SRE,self).__init__(ps_params,str_section)
        
    def update(self,dict_in):
        if self.data == []:
            self.w = dict_in['w'][self.real_imag]
            self.get_legend_info(self.w)
            self.subband_energies = [norm(self.w.get_subband(j).flatten(),2)**2 for j in xrange(self.w.int_subbands)]
        value = np.array([((norm((self.w.get_subband(j) - dict_in['w_n'][0].get_subband(j)).flatten(),2)**2)/
                           self.subband_energies[j]) for j in xrange(self.w.int_subbands)])
        super(SRE,self).update(value)
        
    class Factory:
        def create(self,ps_params,str_section):
            return SRE(ps_params,str_section)
