#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from libtiff import TIFF as tif

class OutputImage(Metric):
    DEFAULT_SLICE = -1
    """
    Class for outputting and image or volume, and allowing a redraw/update.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for OutputImage. See :func:`py_utils.results.metric.Metric.__init__`.

        :param ps_parameters: :class:`py_utils.parameter_struct.ParameterStruct` object which should have 'slice', 'outputformat', and 'outputdirectory' in the :param:`str_section` heading.
        
        """
        super(OutputImage,self).__init__(ps_parameters,str_section)
        self.slice = self.get_val('slice',True, DEFAULT_SLICE)
        self.print_values = 0 #we never want to print array data...
        
    def update(self,dict_in):
        """Takes a 2D or 3D image or volume . If a volume, display the :attr:`self.slice` if specified, otherwise display volume using Myavi. Aggregate images/slices into a volume to view the reconstruction later, or save volumes

        :param dict_in: Input dictionary which contains the referenece to the image/volume data to display and record. 
        """
            if self.slice == DEFAULT_SLICE and dict_in['x_n'].ndim == 3:
                if self.data == []:
                    self.data = [None]
                else:
                    self.data.append(None)    
                self.data[0] = dict_in['x_n']

            else:    
                #updating and storing just each slice of a volume or image
                if dict_in['x_n'].ndim == 3:
                    if self.slice > dict_in['x_n'].shape[2]:
                        ValueError('slice outside of range')
                    else:    
                        self.data.append(dict_in['x_n'][:,:,self.slice])
                else:
                    self.data.append(dict_in['x_n'])
        super(OutputImage,self).update()
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputImage(ps_parameters,str_section)
