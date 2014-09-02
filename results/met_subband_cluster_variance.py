#!/usr/bin/python -tt
import numpy as np
from numpy import log10
from numpy.linalg import norm
from py_utils.results.met_subband_metric import SubbandMetric
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center
from py_utils.section_factory import SectionFactory as sf

import pdb


class SCV(SubbandMetric):
    """
    The subband cluster variance (MSIST-G).
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for SRE.
        """       
        super(SCV,self).__init__(ps_params,str_section)
        self.legend_pos = 'lower center'
        
    def update(self,dict_in):
        if self.data == []:
            self.ls_S_hat_sup = dict_in['ls_S_hat_sup']
            self.get_legend_info(self.ls_S_hat_sup[0], True)
            self.dup_it = dict_in['dup_it']
            self.cluster_norm = sum([self.ls_S_hat_sup[j] for j in self.dup_it])
            self.zeroval = 10E-17
            value = np.array([self.zeroval]*self.ls_S_hat_sup[0].int_subbands)
        else:    
            minus_w_bar_n = dict_in['w_bar_n'][self.real_imag] * (-1)
            ls_w_hat_n = dict_in['ls_w_hat_n'][self.real_imag]
            ls_S_bar_n = [(ls_w_hat_n[j] + minus_w_bar_n) * self.ls_S_hat_sup[j] for j in self.dup_it] #fill in the gaps with w_n
            ls_S_bar_n = [ls_S_bar_n[j].energy() for j in self.dup_it] #compute the variance from differences
            S_bar_n = sum([ls_S_bar_n[j] for j in self.dup_it]) / self.cluster_norm
            value = np.array([self.zeroval]+[np.mean(S_bar_n.get_subband(j)) for j in xrange(1,S_bar_n.int_subbands)])
        super(SCV,self).update(value)
        
    class Factory:
        def create(self,ps_params,str_section):
            return SCV(ps_params,str_section)
