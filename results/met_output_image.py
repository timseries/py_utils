#!/usr/bin/python -tt
import numpy as np
from libtiff import TIFF as tif
import matplotlib.pyplot as plt
import png
from PIL import Image

from py_utils.results.metric import Metric
from py_utils.results.defaults import DEFAULT_SLICE,DEFAULT_IMAGE_EXT

class OutputImage(Metric):
    """
    Class for outputting and image or volume, and allowing a redraw/update.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for OutputImage. See :func:`py_utils.results.metric.Metric.__init__`.

        :param ps_parameters: :class:`py_utils.parameter_struct.ParameterStruct` object which should have 'slice', 'outputformat', and 'outputdirectory' in the :param:`str_section` heading.
        
        """
        super(OutputImage,self).__init__(ps_parameters,str_section)
        #use self.slice=-1 for 'all',otherwise specify a slice number
        self.slice = self.get_val('slice',True, DEFAULT_SLICE)
        self.slices = None
        self.output_extension = self.get_val('outputextension', False, DEFAULT_IMAGE_EXT)
        self.last_frame_only = self.get_val('LastFrameOnly',True) #just the last frame of 'data'
        self.print_values = 0 #we never want to print array data...
        self.has_csv = False #we can't save these to csv format like other metrics
        
    def update(self,dict_in):
        """Takes a 2D or 3D image or volume . If a volume, display the :attr:`self.slice` if specified, otherwise display volume using Myavi. Aggregate images/slices into a volume to view the reconstruction later, or save volumes
        :param dict_in: Input dictionary which contains the referenece to the image/volume data to display and record. 
        """
        value=dict_in[self.key]
        if self.data == []:
            #there always needs to be a dict_in['x'] reference image
            # to compute the input range, for scaling the output range
            self.slices = [slice(0,None,None) if i < 2 else 
                           slice(max(0,min(self.slice,value.shape[i])),None,None) 
                           for i in xrange(value.ndim)]
            self.input_range = np.asarray([np.min(dict_in['x']),
                                           np.max(dict_in['x'])])
        super(OutputImage,self).update(value[self.slices])

    def plot(self):
        if self.data[-1].ndim==2:
            plt.figure(self.figure_number)
            plt.imshow(self.data[-1][self.slices],cmap='gray')

    def save(self,strPath='/home/outputimage'):
        if self.last_frame_only:
            frame_iterator=(len(self.data)-1,self.data[-1])
        else:
            frame_iterator=enumerate(self.data)
            #iterate through the frames    
        for ix,frame in frame_iterator:
            strSavePath = strPath + '_' + str(ix+1) + '.' + self.output_extension
            write_data = frame[self.slices]
            #clip the output range to the input range
            write_data[write_data<self.input_range[0]]=self.input_range[0]
            write_data[write_data>self.input_range[1]]=self.input_range[1]
            #double precision is memory consumptive
            write_data=np.asarray(write_data,dtype='float32')
            if self.output_extension=='png':
                f = open(strSavePath,'wb')
                w = png.Writer(*(write_data.shape[1],write_data.shape[0]),greyscale=True)
                w.write(f,write_data)
                f.close()
            elif self.output_extension=='eps':
                fig = plt.figure()
                ax = plt.Axes(fig,[0,0,1,1])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(write_data,cmap='gray')
                plt.savefig(strSavePath, format="eps",bbox_inches='tight')
            elif self.output_extension=='tif':
                output = tif.open(strSavePath, mode='w')
                #saving as tiff
                if write_data.ndim==3:
                    #axes swapping due to an oddity in the tiff
                    #package, writes 3-D ndarray RowsXColumnsXFrames
                    output.write_image(write_data.swapaxes(2,0).swapaxes(2,1))
                elif write_data.ndim==2:    
                    output.write_image(write_data)
                else:
                    raise ValueError('unsupported # of dims in output.write_image tif')
                output.close()
            else:
                raise ValueError('unsupported extension')
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputImage(ps_parameters,str_section)
