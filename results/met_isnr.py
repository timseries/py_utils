#!/usr/bin/python -tt
from numpy import log10
from numpy.linalg import norm
from py_utils.results.metric import Metric
from py_utils.signal_utilities.sig_utils import noise_gen, crop
from py_utils.section_factory import SectionFactory as sf

class ISNR(Metric):
    """
    Base class for defining a metric
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for ISNR.
        """       
        super(ISNR,self).__init__(ps_params,str_section)
        self.y = None #observation
        self.x = None #ground truth
        self.y_key = self.get_val('comparisony',False)
        if self.y_key == '':
            self.y_key = 'y'
        self.transform_name = self.get_val('transform',False)
        if self.transform_name != '':
            self.transform = sf.create_section(ps_params,
                                               self.transform_name)
        else:
            self.transform = None   
        
    def update(self,dict_in):
        if self.data == []:
            if self.transform != None:
                self.x = (self.transform * dict_in['x']).flatten()
            else:
                if dict_in[self.y_key].shape != dict_in['x'].shape:
                    self.x = crop(dict_in['x'],dict_in['y'].shape).flatten()
                else:
                    self.x = dict_in['x'].flatten()
            self.y = dict_in[self.y_key].flatten()
        if self.transform != None:
            x_n = (self.transform * dict_in['x_n']).flatten()
        else:
            if dict_in[self.y_key].shape != dict_in['x_n'].shape:                
                x_n = crop(dict_in['x_n'],dict_in[self.y_key].shape).flatten()
            else:
                x_n = dict_in['x_n'].flatten()
        value = 10 * log10((norm(self.y - self.x,2)**2)/(norm(x_n - self.x,2)**2))
        super(ISNR,self).update(value)
        
    class Factory:
        def create(self,ps_params,str_section):
            return ISNR(ps_params,str_section)
