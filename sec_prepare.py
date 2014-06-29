#!/usr/bin/python -tt
import numpy as np
from numpy import max as nmax, conj, mean, angle, sqrt, exp
from numpy.fft import fftn, ifftn, fftshift

from py_utils.section import Section
from py_utils.signal_utilities.sig_utils import phase_unwrap
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator_comp import OperatorComp

import matplotlib.pyplot as plt

class Prepare(Section):
    """A configurable set of data preparation routines. 

    Attributes:
        str_type (str): The type of preparation object this is.
        mask_sec_in (str): The name of the mask input section parameters.
    
    """       
    def __init__(self,ps_params,str_section):

        """
        

        """
        super(Prepare,self).__init__(ps_params,str_section)
        self.str_type = self.get_val('preparetype',False)

    def prepare(self,dict_in):
        """Loads observation model parameters into a dictionary, 
        performs the forward model and provides an initial solution.

        Args:
        dict_in (dict): Dictionary which must include the following members:
            'x' (ndarray): The 'ground truth' input signal to be modified.
            '' 
        """
        #build the preparation parameters
        if (self.str_type == 'phasevelocity'):
            mask_sec_in = self.get_val('masksectioninput',False)
            ls_vcorrect_sec_in = self.get_val('vcorrects',False,'',True)
            ls_vcorrect_secs = [sf.create_section(self.get_params(),vcorrect_sec_in)
                                for vcorrect_sec_in in ls_vcorrect_sec_in] 
            #load the mask
            if self.mask_sec_in:
                sec_mask_in = sf.create_section(self.get_params(),mask_sec_in)
                dict_in['mask'] = sec_mask_in.read(dict_in, True)
                sec_bmask_in = sf.create_section(self.get_params(),bmask_sec_in)
                dict_in['boundarymask'] = sec_bmask_in.read(dict_in, True)
            else:
                raise ValueError('need a mask section input')   

            #The frame ordering determines in which direction to compute the 
            #phase differences to obtain positive velocities
            
            frame_order = [0, 1]
            if self.get_val('reverseframeorder'):
                frame_order = [1, 0]
                
            #Fully sampled fourier transform in order to extract phase data
            
            for frame in xrange(2):
                dict_in['x'][:,:,frame]=fftn(fftshift(dict_in['x'][:,:,frame]))
            if self.get_val('extrafftshift',False):
                for frame in xrange(2):
                    dict_in['x'][:,:,frame] = fftshift(dict_in['x'][:,:,frame])
                        
            #Compute phase differences between the two frames
            diff_method = self.get_val('phasedifferencemethod')
            if diff_method == 'conjugateproduct':
                new_x = (dict_in['x'][:,:,frame_order[0]] * 
                         conj(dict_in['x'][:,:,frame_order[1]]))
                theta = angle(new_x)
                magnitude = sqrt(abs(f));
                
            elif diff_method == 'subtraction':
                theta = (dict_in['x'][:,:,frame_order[0]] - 
                         dict_in['x'][:,:,frame_order[2]])
                magnitude = 0.5*(dict_in['x'][:,:,frame_order[1]]
                                 + dict_in['x'][:,:,frame_order[2]])
                new_x = magnitude*exp(1j*theta)
                
            #Do phase unwrapping. This works almost everywhere, except
            #in certain areas where the range of phases exceeds 2*pi.
            #These areas must also be unwrapped with special limits
            #which are determined from the data.
            dict_global_lims = {}
            dict_global_lims['lowerlimit'] = self.get_val('phaselowerlimit')
            dict_global_lims['upperlimit'] = self.get_val('phaseupperlimit')
            dict_global_lims['boundary_mask'] = dict_in['boundarymask']
            dict_global_lims['boundary_upperlimit'] = self.get_val('boundaryphaseupperlimit');
            
            theta = phase_unwrap(theta, dict_global_lims, ls_vcorrect_secs)
            magnitude /= np.max(magnitude)
            new_x = magnitude*exp(1j*theta)

            dict_in['x'] = new_x
            dict_in['dict_global_lims'] = dict_global_lims
            dict_in['ls_vcorrect_secs'] = ls_vcorrect_secs
            
    class Factory:
        def create(self,ps_params,str_section):
            return Prepare(ps_params,str_section)
