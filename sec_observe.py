#!/usr/bin/python -tt
import numpy as np
from numpy import max as nmax, conj, mean, log10, real, angle, abs as nabs, exp
from numpy.linalg import norm
from numpy import asarray as ar
from numpy.random import permutation
from numpy.fft import fftn, ifftn
from scipy.interpolate import griddata
import warnings

from py_utils.section import Section
import py_utils.signal_utilities.sig_utils as su
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift, colonvec, noise_gen, crop_center, pad_center
from py_operators.operator_comp import OperatorComp

from collections import defaultdict

import pdb 
import matplotlib.pyplot as plt

class Observe(Section):
    """
    A configurable observation model. 

    Attributes:
      str_type (str): convolution, convolution_poisson
      Phi (OperatorComp): forward OperatorComp of modalities.
      W (OperatorComp): forward OperatorComp of transforms.

    """       
    def __init__(self,ps_params,str_section):
        super(Observe,self).__init__(ps_params,str_section)
        self.str_type = self.get_val('observationtype',False)
        if self.get_val('modalities',False)!='':
            self.Phi = OperatorComp(ps_params,self.get_val('modalities',False))
            if len(self.Phi.ls_ops)==1: #avoid slow 'eval' in OperatorComp
                self.Phi = self.Phi.ls_ops[0] 
        if self.get_val('transforms',False)!='':
            self.W = OperatorComp(ps_params,self.get_val('transforms',False))
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
        #########################################
        #fetch observation model parameters here#
        #########################################

        if (self.str_type[:11] == 'convolution' or 
            self.str_type == 'compressed_sensing'):
            wrf = self.get_val('wienerfactor',True)
            str_domain = self.get_val('domain',False)
            noise_pars = defaultdict(int) #build a dict to generate the noise
            noise_pars['seed'] = self.get_val('seed',True)
            noise_pars['variance'] = self.get_val('noisevariance', True)
            noise_pars['distribution'] = self.get_val('noisedistribution',False)
            noise_pars['mean'] = self.get_val('noisemean',True)
            noise_pars['interval'] = self.get_val('noiseinterval',True)#uniform
            noise_pars['size'] = dict_in['x'].shape
            dict_in['noisevariance'] = noise_pars['variance']
            
            if self.str_type == 'compressed_sensing':
                noise_pars['complex_noise'] = 1
            if dict_in['noisevariance']>0:
                dict_in['n'] = noise_gen(noise_pars)
            else:
                dict_in['n'] = 0
                
        elif self.str_type=='classification':
            #partition the classification dataset into an 'observed' training set
            #and an unobserved evaluation/test set, and generate features
            dict_in['x_train'] = {}
            dict_in['x_test'] = {}
            dict_in['y_label'] = {}
            dict_in['x_feature'] = {}
            dict_in['n_training_samples'] = 0
            dict_in['n_testing_samples'] = 0
            shuffle=self.get_val('shuffle',True)
            if shuffle:
                shuffleseed=self.get_val('shuffleseed',True)
            training_proportion = self.get_val('trainingproportion',True)    
            classes = dict_in['x'].keys()
            #partition and generate numeric class labels 
            for _class_index,_class in enumerate(classes):
                class_size = len(dict_in['x'][_class])
                training_size = int(training_proportion*class_size)
                dict_in['n_training_samples'] += training_size
                dict_in['n_testing_samples'] += class_size-training_size
                if shuffle:
                    np.random.seed(shuffleseed)
                    indices = np.random.permutation(class_size)
                else:
                    indices = np.array(range(class_size),dtype='uint16')
                dict_in['x_train'][_class]=indices[:training_size]
                dict_in['x_test'][_class]=indices[training_size:]
                dict_in['y_label'][_class]=_class_index
        else:    
            raise ValueError('unsupported observation model')     
        ################################################
        #compute the forward model and initial estimate#
        ################################################
        if self.str_type == 'convolution':
                H = self.Phi
                H.set_output_fourier(False)
                dict_in['Hx'] = H * dict_in['x']
                dict_in['y'] = dict_in['Hx']+dict_in['n']
                #regularized Wiener filtering in Fourier domain
                H.set_output_fourier(True)
                dict_in['x_0'] = real(ifftn(~H * dict_in['y'] /
                                            (H.get_spectrum_sq() + 
                                             wrf * noise_pars['variance'])))
                H.set_output_fourier(False)
            #compute bsnr    
                self.compute_bsnr(dict_in,noise_pars)
        elif self.str_type == 'convolution_downsample':
                Phi = self.Phi
                #this order is important in the config file
                D = Phi.ls_ops[1]
                H = Phi.ls_ops[0]
                H.set_output_fourier(False)
                dict_in['Phix'] = Phi * dict_in['x']
                dict_in['Hx'] = dict_in['Phix']
                #the version of y without downsampling
                dict_in['Hxpn'] = H * dict_in['x'] + dict_in['n']
                dict_in['DHxpn'] = np.zeros((D*dict_in['Hxpn']).shape)
                if dict_in['n'].__class__.__name__ == 'ndarray':
                    dict_in['n'] = D * dict_in['n']
                dict_in['y'] = dict_in['Hx']+dict_in['n']
                DH = fftn(Phi*nd_impulse(dict_in['x'].shape))
                DHt = conj(DH)
                Hty=fftn(D*(~Phi * dict_in['y']))
                HtDtDH=np.real(DHt*DH)
                dict_in['x_0'] = ~D*real(ifftn(Hty /
                                               (HtDtDH + 
                                                wrf * noise_pars['variance'])))
                #optional interpolation
                xdim=dict_in['x'].ndim
                if self.get_val('interpinitialsolution',True) and xdim < 3:
                    xshp=dict_in['x'].shape
                    grids=np.mgrid[[slice(0,xshp[j]) for j in xrange(xdim)]]
                    grids = tuple([grids[i] for i in xrange(grids.shape[0])])
                    sampled_coords = np.mgrid[[slice(D.offset[j],xshp[j],D.ds_factor[j]) 
                                               for j in xrange(xdim)]]
                    values = dict_in['x_0'][[coord.flatten() for coord in sampled_coords]]
                    points = np.vstack([sampled_coords[i, Ellipsis].flatten() 
                                        for i in xrange(sampled_coords.shape[0])]).transpose() #pts to interp
                    interp_vals = griddata(points,values,grids,method='cubic',fill_value=0.0)
                    dict_in['x_0']=interp_vals
                self.compute_bsnr(dict_in,noise_pars)
                
        elif self.str_type == 'convolution_poisson':
            dict_in['mp'] = self.get_val('maximumphotonspervoxel',True)
            dict_in['b'] = self.get_val('background', True)
            H = self.Phi
            if str_domain == 'fourier':
                H.set_output_fourier(False) #return spatial domain object
                orig_shape = dict_in['x'].shape
                Hspec = np.zeros(orig_shape)
                dict_in['r'] = H * dict_in['x']
                k = dict_in['mp'] / nmax(dict_in['r'])
                dict_in['r'] = k * dict_in['r']
                #normalize the output image to have the same
                #maximum photon count as the ouput image
                dict_in['x'] = k * dict_in['x']
                dict_in['x'] = crop_center(dict_in['x'],dict_in['r'].shape).astype('float32')
                #the spatial domain measurements, before photon counts
                dict_in['fb'] = dict_in['r'] + dict_in['b']
                #lambda of the poisson distn
                noise_pars['ary_mean'] = dict_in['fb']
                #specifying the poisson distn
                noise_distn2 = self.get_val('noisedistribution2',False)
                noise_pars['distribution'] = noise_distn2
                #generating quantized (uint16) poisson measurements
                dict_in['y'] = (noise_gen(noise_pars)
                                ).astype('uint16').astype('int32')
            elif str_domain == 'evaluation': #are given the observation, which is stored in 'x'
                dict_in['y'] = dict_in.pop('x')
            else:
                raise Exception('domain not supported: ' + str_domain)
            dict_in['x_0'] = ((~H) * (dict_in['y'])).astype(dtype='float32')
            dict_in['y_padded'] = pad_center(dict_in['y'],dict_in['x_0'].shape)
            
        elif self.str_type == 'compressed_sensing':
            Fu = self.Phi
            dict_in['Hx'] = Fu * dict_in['x']
            dict_in['y'] = dict_in['Hx'] + dict_in['n']
            dict_in['x_0'] = (~Fu) * dict_in['y']
            dict_in['theta_0'] = angle(dict_in['x_0'])
            dict_in['theta_0'] = su.phase_unwrap(dict_in['theta_0'],dict_in['dict_global_lims'],dict_in['ls_local_lim_secs'])
            dict_in['theta_0'] *= dict_in['mask']
            dict_in['magnitude_0'] = nabs(dict_in['x_0'])
            dict_in['x_0'] = dict_in['magnitude_0']*exp(1j*dict_in['theta_0'])
            self.compute_bsnr(dict_in,noise_pars)
        #store the wavelet domain version of the ground truth
        if np.iscomplexobj(dict_in['x']):
            dict_in['w']  = [self.W * dict_in['x'].real,self.W * dict_in['x'].imag]
        else:
            dict_in['w']  = [self.W * dict_in['x']]
            
    def compute_bsnr(self,dict_in,noise_pars):
        Hx = dict_in['Hx'].flatten()
        sig_sq = noise_pars['variance']
        if sig_sq == 0:
            dict_in['bsnr']  = np.inf
        else:    
            dict_in['bsnr'] = 10*log10((norm(Hx-mean(Hx),ord=2)**2) /
                                       (Hx.size * sig_sq))
        print 'observed with BSNR: ' + str(dict_in['bsnr'])
        

    class Factory:
        def create(self,ps_params,str_section):
            return Observe(ps_params,str_section)
