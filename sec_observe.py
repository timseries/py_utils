#!/usr/bin/python -tt
from py_utils.section import Section
from numpy import max as nmax, conj, mean, log10
from numpy.linalg import norm
from numpy import asarray as ar
from numpy.fft import fftn, ifftn
from py_operators.operator_comp import OperatorComp
import warnings
import numpy as np
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift, colonvec, noise_gen, crop

class Observe(Section):
    """
    A configurable observation model. 

    Attributes:
      str_type (str): convolution, convolution_poisson
      H (OperatorComp): forward OperatorComp of operators.
      W (OperatorComp): forward OperatorComp of transforms.

    """       
    def __init__(self,ps_params,str_section):
        super(Observe,self).__init__(ps_params,str_section)
        self.str_type = self.get_val('observationtype',False)
        self.H = OperatorComp(ps_params,
                              self.get_val('modalities',False))
        if len(self.H.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.H = self.H.ls_ops[0] 
        self.W = OperatorComp(ps_params,
                              self.get_val('transforms',False))
        if len(self.W.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.W = self.W.ls_ops[0] 
            
    def observe(self,dict_in):
        """
        Loads observation model parameters into a dictionary, 
        performs the forward model and provides an initial solution.

        Args:
        dict_in (dict): Dictionary which will be overwritten with 
        all of the observation model parameters, forward model 
        observation 'y', and initial estimate 'x_0'.
        """
        warnings.simplefilter("ignore",np.ComplexWarning)
        #build the observation model parameters
        if self.str_type == 'convolution' or \
          self.str_type == 'convolution_poisson':
            H = self.H
            wrf = nmax(self.get_val('wienerfactor',True),0.001)
            str_domain = self.get_val('domain',False)
            noise_pars = {} #build a dict to generate the noise
            noise_pars['seed'] = self.get_val('seed',True)
            noise_pars['variance'] = self.get_val('noisevariance',
                                                  True)
            noise_pars['distribution'] = self.get_val('noisedistribution',False)
            noise_pars['mean'] = self.get_val('noisemean',True)
            noise_pars['interval'] = self.get_val('noiseinterval',True)#uniform
            noise_pars['size'] = dict_in['x'].shape
            dict_in['noisevariance'] = noise_pars['variance']
            dict_in['n'] = noise_gen(noise_pars)
            dict_in['nhat'] = fftn(noise_gen(noise_pars))
        else:
            ValueError('unsupported observation model')     
        #compute the forward model and initial estimate
        if self.str_type == 'convolution':
            if not H.spatial:
                H.output_fourier = 0
                dict_in['Hx'] = H * dict_in['x']
                dict_in['y'] = dict_in['Hx']+dict_in['n']
                #regularized Wiener filtering in Fourier domain
                H.output_fourier = 1
                dict_in['x_0'] = ifftn(H.t() * dict_in['y'] /
                                       (H.get_spectrum_sq() + 
                                        wrf * noise_pars['variance']))
                H.output_fourier = 0
            else:
                ValueError('spatial domain convolution not supported')
            #compute bsnr    
            Hx = dict_in['Hx'].flatten()
            sig_sq = noise_pars['variance']
            dict_in['bsnr'] = 10*log10((norm(Hx-mean(Hx),ord=2)**2) /
                                       (Hx.size * sig_sq))
            print 'observed with BSNR: ' + str(dict_in['bsnr'])
        elif self.str_type == 'convolution_poisson':
            dict_in['mp'] = self.get_val('maximumphotonspervoxel',True)
            dict_in['b'] = self.get_val('background', True)
            if str_domain == 'fourier':
                orig_shape = dict_in['x'].shape
                Hspec = np.zeros(orig_shape)
                dict_in['r'] = H * dict_in['x'] #Direct...
                dict_in['w'] = self.W * dict_in['x']
                dict_in['w'].flatten()
                dict_in['Hxhat'] = fftn(dict_in['r'])
                k = dict_in['mp'] / nmax(dict_in['r'])
                dict_in['r'] = k * dict_in['r']
                dict_in['x'] = k * dict_in['x']
                dict_in['fb'] = dict_in['r'] + dict_in['b']
                dict_in['x'] = crop(dict_in['x'],dict_in['r'].shape)
                dict_in['x_f'] = fftn(dict_in['x'])
                noise_pars['ary_mean'] = dict_in['fb']
                noise_distn2 = self.get_val('noisedistribution2',False)
                noise_pars['distribution'] = noise_distn2
                dict_in['y'] = (noise_gen(noise_pars)
                                ).astype('uint16').astype('int32')
                #inverse filtering in fourier domain to find initial solution
                Hspec[tuple([Hspec.shape[i]/2 
                             for i in np.arange(Hspec.ndim)])]=1.0
                Hspec = fftn(H * Hspec)
                Hty = (~H) * dict_in['y']
                Hty_crop_hat = fftn(crop(Hty,dict_in['x'].shape))
                x0 = np.real(ifftn(Hty_crop_hat/
                                   (conj(Hspec)*Hspec
                                    + wrf * noise_pars['variance'])))
                dict_in['x_0'] = np.zeros(orig_shape)
                ary_small = ar([(dict_in['x_0'].shape[i] 
                                         - x0.shape[i])/2 + 1 
                                         for i in np.arange(x0.ndim)])
                ary_large = ar([ary_small[i] + x0.shape[i] - 1
                                for i in np.arange(x0.ndim)])
                slices=colonvec(ary_small,ary_large)
                dict_in['x_0'][slices]=x0
                dict_in['y_padded'] = np.zeros(orig_shape)
                dict_in['y_padded'][slices] = dict_in['y']
                #simple adjoint to find initial solutino
                #dict_in['x_0'] = ((~H) * (dict_in['y'])).astype(dtype='float32')
                #dict_in['x_0'] = np.real(ifftn(fftn(~H * dict_in['y']) / \
                #(conj(H.get_spectrum()) * H.get_spectrum() + wrf * noise_pars['variance'])))
            else:
                raise Exception('spatial convolution not supported') 
            
        elif self.str_type == 'compressedsensing':
            raise Exception('cs observation not supported yet')    
        else:
            raise Exception('no observation type: ' + self.str_type)

    class Factory:
        def create(self,ps_params,str_section):
            return Observe(ps_params,str_section)
