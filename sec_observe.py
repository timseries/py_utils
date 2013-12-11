#!/usr/bin/python -tt
from py_utils.section import Section
from numpy import max as nmax, conj, mean, log10
from numpy.linalg import norm
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
            wrf = nmax(self.get_val('wienerfactor',True),0.001)
            H = self.H
            noise_pars = {}
            noise_pars['seed'] = self.get_val('seed',True)
            noise_pars['variance'] = self.get_val('noisevariance',True)
            noise_pars['distribution'] = self.get_val('noisedistribution',False)
            noise_pars['mean'] = self.get_val('noisemean',True)
            noise_pars['interval'] = self.get_val('noiseinterval',True)#uniform
            noise_pars['size'] = dict_in['x'].shape
            if str_domain == 'fourier':
                dict_in['n'] = fftn(noise_gen(noise_pars))
                dict_in['Hxhat'] = H * dict_in['x']
                dict_in['Hx'] = ifftn(dict_in['Hxhat'])
                dict_in['yhat'] = dict_in['Hxhat'] + dict_in['n']
                dict_in['y'] = ifftn(dict_in['yhat'])
                #inverse filtering in fourier domain
                dict_in['x_0'] = ifftn((~H * dict_in['y']) / \
                  (conj(H.get_spectrum()) * H.get_spectrum() + wrf * noise_pars['variance']))
            else:
                raise Exception('spatial domain convolution not supported')    
            dict_in['bsnr'] = 10 * log10(norm(dict_in['Hx'].flatten() - mean(dict_in['Hx'].flatten()),ord=2)**2 / \
                                         (dict_in['Hx'].size * noise_pars['variance']))
            print 'made blurry observation with BSNR: ' + str(dict_in['bsnr'])
        elif self.str_observation_type == 'compressedsensing':
            raise Exception('cs observation not supported yet')    
        else:
            raise Exception('no observation type: ' + self.str_observation_type)    
        
    class Factory:
        def create(self,ps_parameters,str_section):

            return Observe(ps_parameters,str_section)
