#!/usr/bin/python -tt
import numpy as np
from numpy import log10
from numpy.linalg import norm
from py_utils.results.metric import Metric
from py_utils.signal_utilities.sig_utils import noise_gen, crop_center
from py_utils.section_factory import SectionFactory as sf


class Matrix(Metric):
    """
    The base class for writing a 2d matrix to a csv file with columns  row0....rowN...col0...colN
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for SRE.
        """       
        super(Matrix,self).__init__(ps_params,str_section)
        self.last_frame_only = self.get_val('lastframeonly',True) 
    def update(self,value):
        super(Matrix,self).update(value)
        
    def save(self,strPath='/home/outputimage/'):
        self.save_csv(strPath)

    class Factory:
        def create(self,ps_params,str_section):
            return Matrix(ps_params,str_section)
