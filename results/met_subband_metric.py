#!/usr/bin/python -tt
import numpy as np
from numpy import log10
from numpy.linalg import norm
from py_utils.results.metric import Metric
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center
from py_utils.section_factory import SectionFactory as sf


class SubbandMetric(Metric):
    """
    The base class for computing/displaying subband-based metrics (to commonalize the plotting and storage).
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for SRE.
        """       
        super(SubbandMetric,self).__init__(ps_params,str_section)
        self.real_imag = self.get_val('realorimag',True)
        
    def update(self,value):
        super(SubbandMetric,self).update(value)
        
    def save(self,strPath='/home/outputimage/'):
        self.save_csv(strPath)

    def get_legend_info(self, ws_in, lowpass=True):
        '''populate legend entries
        '''
        legend_labels_temp = [ws_in.lev_ori_from_subband(j,ori_degrees=True) for j in xrange(1, ws_in.int_subbands)]
        self.legend_labels = [r'$s$: ' + str(label_el[0]+1)+r', $\theta$: '+ str(label_el[1]) 
                              for label_el in legend_labels_temp]
        if lowpass==True:
            self.legend_labels = ['Lowpass'] + self.legend_labels
        self.legend_cols = label_el[0]+1 #the number of columns to display in the legend
        
    class Factory:
        def create(self,ps_params,str_section):
            return SubbandMetric(ps_params,str_section)
