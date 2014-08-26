#!/usr/bin/python -tt
import numpy as np
from numpy import log10
from numpy.linalg import norm
from py_utils.results.metric import Metric
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center
from py_utils.section_factory import SectionFactory as sf


class SRE(Metric):
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
            self.w = dict_in['w']
            self.subband_energies = [norm(self.w.get_subband(j).flatten(),2)**2 for j in xrange(self.w.int_subbands)]
            legend_labels_temp = [self.w.lev_ori_from_subband(j,ori_degrees=True) for j in xrange(1,self.w.int_subbands)]
            self.legend_labels = ['Lowpass'] + [r'$l$: ' + str(label_el[0]+1)+r', $\theta$: '+ str(label_el[1]) 
                                  for label_el in legend_labels_temp]
            self.legend_cols = label_el[0]+1
            
        value = np.array([((norm((self.w.get_subband(j) - dict_in['w_n'][0].get_subband(j)).flatten(),2)**2)/
                           self.subband_energies[j]) for j in xrange(self.w.int_subbands)])
        super(SRE,self).update(value)
        
    def save(self,strPath='/home/outputimage/'):
        self.save_csv(strPath)
            
    class Factory:
        def create(self,ps_params,str_section):
            return SRE(ps_params,str_section)
