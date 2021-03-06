#!/usr/bin/python -tt
import numpy as np
from numpy import max as nmax, conj, mean, angle, sqrt, exp, abs as nabs
from numpy.fft import fftn, ifftn, fftshift

from py_utils.section import Section
from py_utils.signal_utilities.sig_utils import phase_unwrap, pad_center, crop_center
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator_comp import OperatorComp

import matplotlib.pyplot as plt

import pdb

class Preprocess(Section):
    """A configurable set of data preprocessing routines. 

    Attributes:
        str_type (str): The type of preprocessing object this is.
        mask_sec_in (str): The name of the mask input section parameters.
    
    """       
    def __init__(self,ps_params,str_section):
        """
        """
        super(Preprocess,self).__init__(ps_params,str_section)
        self.str_type = self.get_val('preprocesstype',False)

    def preprocess(self,dict_in):
        """Loads observation model parameters into a dictionary, 
        performs the forward model and provides an initial solution.

        Args:
        dict_in (dict): Dictionary which must include the following members:
            'x' (ndarray): The 'ground truth' input signal to be modified.
        """
        #build the preprocessing parameters
        if (self.str_type == 'brainwebmri'):
            #need to pad/crop the input data for wavelet processing
            swap_axes = self.get_val('swapaxes', True)
            if swap_axes.__class__.__name__ == 'ndarray':
                dict_in['x'] = dict_in['x'].swapaxes(swap_axes[0],swap_axes[1])
            input_shape = dict_in['x'].shape
            
            #cropping
            new_shape = self.get_val('newshape', True)
            if new_shape.__class__.__name__ == 'ndarray':
                new_shape = tuple(new_shape)
            #figure out what to crop, if anything
            if np.any(new_shape<input_shape):
                crop_shape = np.min(np.vstack((new_shape,input_shape)),axis=0)
                dict_in['x'] = crop_center(dict_in['x'], crop_shape)
            else:
                crop_shape = input_shape
            #padding    
            if np.any(new_shape>crop_shape):
                pad_shape = np.max(np.vstack((new_shape,crop_shape)),axis=0)
                dict_in['x'] = pad_center(dict_in['x'], pad_shape)
        # elif (self.str_type == 'superresolution'):
        #     #need to crop edge of image to make results compatible with the literature
            
        elif (self.str_type == 'phasevelocity'):
            mask_sec_in = self.get_val('masksectioninput',False)
            bmask_sec_in = self.get_val('boundarymasksectioninput',False)
            ls_local_lim_sec_in = self.get_val('vcorrects',False)
            if ls_local_lim_sec_in.__class__.__name__ == 'str' and ls_local_lim_sec_in:
                ls_local_lim_sec_in = [ls_local_lim_sec_in]
            ls_local_lim_secs = []    
            if ls_local_lim_sec_in:
                ls_local_lim_secs = [sf.create_section(self.get_params(),local_lim_sec_in)
                                     for local_lim_sec_in in ls_local_lim_sec_in] 
                ls_local_lim_secs = [{'phaselowerlimit' : local_lim.get_val('phaselowerlimit', True),
                                      'phaseupperlimit' : local_lim.get_val('phaseupperlimit', True), 
                                      'regionupperleft' : local_lim.get_val('regionupperleft', True), 
                                      'regionlowerright' : local_lim.get_val('regionlowerright', True)} 
                                      for local_lim in ls_local_lim_secs]
            #load the mask
            if mask_sec_in != '':
                sec_mask_in = sf.create_section(self.get_params(),mask_sec_in)
                dict_in['mask'] = np.asarray(sec_mask_in.read(dict_in, True), dtype='bool')
            else:
                dict_in['mask'] = True
                
            if bmask_sec_in!='':    
                sec_bmask_in = sf.create_section(self.get_params(),bmask_sec_in)
                dict_in['boundarymask'] = np.asarray(sec_bmask_in.read(dict_in, True), dtype='bool')
            else:
                dict_in['boundarymask'] = np.asarray(np.zeros(dict_in['x'][:,:,0].shape),dtype='bool')

            if self.get_val('nmracquisition',True): #compute phase from lab measurement
                #The frame ordering determines in which direction to compute the 
                #phase differences to obtain positive velocities

                frame_order = [0, 1]
                if self.get_val('reverseframeorder'):
                    frame_order = [1, 0]
                #Fully sampled fourier transform in order to extract phase data
                for frame in xrange(2):
                    dict_in['x'][:,:,frame]=fftn(fftshift(dict_in['x'][:,:,frame]))
                if self.get_val('extrafftshift',True):
                    for frame in xrange(2):
                        dict_in['x'][:,:,frame] = fftshift(dict_in['x'][:,:,frame])

                #Compute phase differences between the two frames
                diff_method = self.get_val('phasedifferencemethod')
                if diff_method == 'conjugateproduct':
                    new_x = (dict_in['x'][:,:,frame_order[1]] * 
                             conj(dict_in['x'][:,:,frame_order[0]]))
                    theta = angle(new_x)
                    theta +=np.max(np.abs(theta))
                    # theta /= np.max(np.abs(theta))
                    # theata *= np.pi*2
                    magnitude = sqrt(abs(new_x))

                elif diff_method == 'subtraction':
                    theta = (angle(dict_in['x'][:,:,frame_order[1]]) - 
                             angle(dict_in['x'][:,:,frame_order[0]]))
                    magnitude = 0.5*(np.abs(dict_in['x'][:,:,frame_order[0]])
                                     + np.abs(dict_in['x'][:,:,frame_order[1]]))
                # if self.get_val('reverseframeorder'):
                #     theta = -theta
                #     theta+=np.abs(np.min(theta))
                new_x = magnitude*exp(1j*theta)
                    
            else: #synthetic data
                theta = angle(dict_in['x'])
                magnitude = nabs(dict_in['x'])

            #Do phase unwrapping. This works almost everywhere, except
            #in certain areas where the range of phases exceeds 2*pi.
            #These areas must also be unwrapped with special limits
            #which are determined from the data.
            dict_global_lims = {}
            dict_global_lims['lowerlimit'] = self.get_val('phaselowerlimit',True)
            dict_global_lims['upperlimit'] = self.get_val('phaseupperlimit',True)
            dict_global_lims['boundary_mask'] = dict_in['boundarymask']
            dict_global_lims['boundary_upperlimit'] = self.get_val('boundaryphaseupperlimit',True)
            dict_global_lims['boundaryoverlapvcorrects'] = self.get_val('boundaryoverlapvcorrects',True)
            
            theta = phase_unwrap(theta, dict_global_lims, ls_local_lim_secs)
            magnitude /= np.max(nabs(magnitude))
            dict_in['x'] = magnitude*exp(1j*theta)
            dict_in['theta'] = dict_in['mask'] * theta
            dict_in['magnitude'] = magnitude
            dict_in['dict_global_lims'] = dict_global_lims
            dict_in['ls_local_lim_secs'] = ls_local_lim_secs

    class Factory:
        def create(self,ps_params,str_section):
            return Preprocess(ps_params,str_section)
