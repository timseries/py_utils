#!/usr/bin/python -tt
from py_utils.section import Section
from numpy import max as nmax, conj
from numpy.fft import fftn, ifftn
from py_operators.operator_comp import OperatorComp

from py_utils.signal_utilities.sig_utils import noise_gen

class Observe(Section):
    """
    Observe class, 
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Observe.
        """       
        super(Observe,self).__init__(ps_parameters,str_section)
        self.str_observation_type = self.get_val('observationtype',False)
        self.H = OperatorComp(ps_parameters,self.get_val('modalities',False))
        self.W = OperatorComp(ps_parameters,self.get_val('transforms',False))
        
    def observe(self,dict_in):
        """
        Append an input dictionary object (dict_in) with the measurment variables
        Provides observation from ground truth, and initial estimate x_0
        """
        if self.str_observation_type == 'convolution':
            str_domain = self.get_val('domain',False)
            dbl_wrf = nmax(self.get_val('wienerfactor',True),0.001)
            H = self.H
            noise_pars = {}
            noise_pars['seed'] = self.get_val('seed',True)
            noise_pars['variance'] = self.get_val('noisevariance',True)
            noise_pars['distribution'] = self.get_val('noisedistribution',False)
            noise_pars['mean'] = self.get_val('noisemean',True)
            noise_pars['interval'] = self.get_val('noiseinterval',True)#uniform
            noise_pars['size'] = dict_in['x'].shape
            if str_domain == 'Fourier':
                dict_in['n'] = fftn(noise_gen(noise_pars))
                dict_in['yhat'] = H * dict_in['x'] + dict_in['n']
                dict_in['y'] = ifftn(dict_in['yhat'])
                #inverse filtering in fourier domain
                dict_in['x_0'] = ifftn(~H * dict_in['y']) / \
                  (conj(H.get_spectrum()) * H.get_spectrum() + dbl_wrf * noise_pars['variance'])
            else:
                raise Exception('spatial domain convolution not supported')    
        elif self.str_observation_type == 'compressedsensing':
            raise Exception('cs observation not supported yet')    
        else:
            raise Exception('no observation type: ' + self.str_observation_type)    
        
    class Factory:
        def create(self,ps_parameters,str_section):

            return Observe(ps_parameters,str_section)
