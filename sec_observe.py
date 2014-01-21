#!/usr/bin/python -tt
from py_utils.section import Section
from numpy import max as nmax, conj, mean, log10
from numpy.linalg import norm
from numpy.fft import fftn, ifftn
from py_operators.operator_comp import OperatorComp
import warnings
import numpy as np
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift, colonvec, noise_gen, crop
#for debug
from py_utils.helpers import numpy_to_mat
import pdb

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
        if len(self.H.ls_operators)==1: #if we just have one transform
            self.H = self.H.ls_operators[0] 
        self.W = OperatorComp(ps_parameters,self.get_val('transforms',False))
        if len(self.W.ls_operators)==1: #assume we just have one transform
            self.W = self.W.ls_operators[0] 
    def observe(self,dict_in):
        """
        Append an input dictionary object (dict_in) with the measurment variables
        Provides observation from ground truth, and initial estimate x_0
        """
        warnings.simplefilter("ignore",np.ComplexWarning)
        if self.str_observation_type == 'convolution' or \
          self.str_observation_type == 'convolution_poisson':
            H = self.H
            wrf = nmax(self.get_val('wienerfactor',True),0.001)
            str_domain = self.get_val('domain',False)
            noise_pars = {}
            noise_pars['seed'] = self.get_val('seed',True)
            noise_pars['variance'] = self.get_val('noisevariance',True)
            dict_in['noisevariance'] = self.get_val('noisevariance',True)
            noise_pars['distribution'] = self.get_val('noisedistribution',False)
            noise_pars['mean'] = self.get_val('noisemean',True)
            noise_pars['interval'] = self.get_val('noiseinterval',True)#uniform
            noise_pars['size'] = dict_in['x'].shape
            dict_in['n'] = noise_gen(noise_pars)
            dict_in['nhat'] = fftn(noise_gen(noise_pars))
        if self.str_observation_type == 'convolution':
            if str_domain == 'fourier':
                dict_in['Hxhat'] = H * dict_in['x']
                dict_in['Hx'] = ifftn(dict_in['Hxhat'])
                dict_in['yhat'] = dict_in['Hxhat'] + dict_in['nhat']
                dict_in['y'] = ifftn(dict_in['yhat'])
                #inverse filtering in fourier domain
                dict_in['x_0'] = ifftn(fftn(~H * dict_in['y']) / \
                  (conj(H.get_spectrum()) * H.get_spectrum() + wrf * noise_pars['variance']))
            else:
                raise Exception('spatial domain convolution not supported')    
            dict_in['bsnr'] = 10 * log10(norm(dict_in['Hx'].flatten() - mean(dict_in['Hx'].flatten()),ord=2)**2 / \
                                         (dict_in['Hx'].size * noise_pars['variance']))
            print 'made blurry observation with BSNR: ' + str(dict_in['bsnr'])
        elif self.str_observation_type == 'convolution_poisson':
            dict_in['mp'] = self.get_val('maximumphotonspervoxel', True)
            dict_in['b'] = self.get_val('background', True)
            if str_domain == 'fourier':
                orig_shape = dict_in['x'].shape
                Hspec = np.zeros(orig_shape)
                dict_in['r'] = H * dict_in['x'] #Direct...

                dict_in['w'] = self.W * dict_in['x']
                dict_in['w'].flatten()
                
                print 'max w: ' + str(np.max(np.abs(dict_in['w'].ws_vector)))
                print 'min w: ' + str(np.min(np.abs(dict_in['w'].ws_vector)))
                #pdb.set_trace()
                #dict_in['w'].flatten()
                #numpy_to_mat(dict_in['w'].ws_vector,'/home/tim/repos/py_solvers/applications/deconvolution_challenge/ws_gt_vector_py.mat','ws_gt_vector_py')    
                #print 'done storing wavelet coeffs'
                #save this result for comparison with matlab implementation...
                dict_in['Hxhat'] = fftn(dict_in['r'])
                k = dict_in['mp'] / nmax(dict_in['r'])
                dict_in['r'] = k * dict_in['r']
                dict_in['x'] = k * dict_in['x']
                dict_in['fb'] = dict_in['r'] + dict_in['b']
                dict_in['x'] = crop(dict_in['x'],dict_in['r'].shape)
                dict_in['x_f'] = fftn(dict_in['x'])
                noise_pars['ary_mean'] = dict_in['fb']
                noise_pars['distribution'] = self.get_val('noisedistribution2',False)
                dict_in['y'] = noise_gen(noise_pars).astype(dtype='uint16').astype(dtype='int32')
                #inverse filtering in fourier domain to find initial solution
                Hspec[tuple([Hspec.shape[i]/2 for i in np.arange(Hspec.ndim)])]=1.0
                Hspec = fftn(H * Hspec)
                #Hty = (~H) * dict_in['y']
                #Hty_crop_hat = fftn(crop(Hty,dict_in['x'].shape))
                #x0 = np.real(ifftn(Hty_crop_hat/(conj(Hspec)*Hspec + wrf * noise_pars['variance'])))

                #dict_in['x_0'] = np.zeros(orig_shape)
                #ary_small = np.asarray([(dict_in['x_0'].shape[i] - x0.shape[i])/2 + 1 for i in np.arange(x0.ndim)])
                #ary_large = np.asarray([ary_small[i] + x0.shape[i] - 1 for i in np.arange(x0.ndim)])
                #slices=colonvec(ary_small,ary_large)
                #dict_in['x_0'][slices]=x0
                #simple adjoint to find initial solutino
                #dict_in['x_0'] = ((~H) * (dict_in['y'])).astype(dtype='float32')
                dict_in['x_0'] = np.real(ifftn(fftn(~H * dict_in['y']) / \
                  (conj(H.get_spectrum()) * H.get_spectrum() + wrf * noise_pars['variance'])))
            else:
                raise Exception('spatial domain convolution not supported')    
            
        elif self.str_observation_type == 'compressedsensing':
            raise Exception('cs observation not supported yet')    
        else:
            raise Exception('no observation type: ' + self.str_observation_type)    

    class Factory:
        def create(self,ps_parameters,str_section):
            return Observe(ps_parameters,str_section)
