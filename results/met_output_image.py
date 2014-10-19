#!/usr/bin/python -tt
import numpy as np
from libtiff import TIFF as tif
#headless imshow
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import png
from PIL import Image
from mpldatacursor import datacursor
import os
import copy

from py_utils.results.metric import Metric
from py_utils.results.defaults import DEFAULT_SLICE,DEFAULT_IMAGE_EXT

import pdb

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
        #just the last 'frame' of image data, where frames are indexed by iteration, for legacy support (superceded by .update_once)
        self.last_frame_only = self.get_val('lastframeonly',True) 
        self.mask_color = self.get_val('maskcolor',True) #3-element rgb array 
        self.mask_key = self.get_val('maskkey',False) #supplies the same-sized array-- wherethe mask vals are
        self.im_range_key = self.get_val('imrangekey',False) 
        if not self.im_range_key:
            self.im_range_key = 'x'
        self.print_values = 0 #we never want to print array data...
        self.has_csv = False #we can't save these to csv format like other metrics
        self.cmap = self.get_val('colormap',False,'gray')
        self.cmap = copy.copy(plt.cm.get_cmap(self.cmap)) #arbitrary cmap
        #mask related code
        self.mask = None
        if self.mask_key !='':
            if self.mask_color.__class__.__name__ != 'ndarray': #assume 0 value here
                self.mask_color = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                self.mask_color = np.hstack((self.mask_color, 1.0))
            self.cmap._init()
            self.cmap._lut[-1,:] = self.mask_color
            self.save_key = self.key + '_maskedby_' + self.mask_key #dont overwrite the original unmasked data
            
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
            if not dict_in.has_key(self.im_range_key):
                im_range_key='y'
            else: #no ground truth, default to observation for input range
                im_range_key=self.im_range_key
            self.input_range = np.asarray([np.min(dict_in[im_range_key]),
                                           np.max(dict_in[im_range_key])])
            if self.mask_key != '' and self.mask_key in dict_in:
                self.mask = dict_in[self.mask_key][self.slices] == 0 #need to flip the 0's to 1's (invalid)
        update_val = value[self.slices]
        if self.mask is not None: #add a mask channel
            update_val = np.ma.array(update_val, mask = self.mask)
        super(OutputImage,self).update(update_val)

    def plot(self):
        if self.data[-1].ndim==2:
            plt.figure(self.figure_number)
            plt.imshow(self.data[-1][self.slices],cmap=self.cmap, 
                       vmin=self.input_range[0],vmax=self.input_range[1])#,interpolation="none")
            if self.get_val('colorbar',True):
                cb = plt.colorbar()
                # cb.set_clim(self.input_range[0],self.input_range[1])
            datacursor(display='single')

    def save(self,strPath='/home/outputimage/'):
        if len(self.data)==0:
            return
        if self.last_frame_only or self.update_once:
            frame_iterator=[('',self.data[-1])]
            ix_offset=''
        else:
            frame_iterator=enumerate(self.data)
            #iterate through the frames    
            #find the correct index offset, given the current path (save_often mode)
            #strPath should contain everything except the current iterate
            
            files_enumerated = enumerate(os.walk(os.path.dirname(strPath)))
            base_name=os.path.basename(strPath)
            files,dir_info=files_enumerated.next()
            if self.save_often:
                #figure out where the last save was, since we're only keeping
                #one iterate in memory at once
                ix_offset=len([file_name for file_name in dir_info[2] if base_name in file_name])
            else:    
                #we're keeping all iterates in memory, so start at 0
                ix_offset=0
        for ix,frame in frame_iterator:
            strSavePath = strPath + str(ix_offset+ix) + '.' + self.output_extension
            write_data = frame[self.slices]
            #clip the output range to the input range
            write_data[write_data<self.input_range[0]]=self.input_range[0]
            write_data[write_data>self.input_range[1]]=self.input_range[1]
            #shift in the case of a negative lower limit (can't have negative intensities)
            if self.input_range[0]<0:
                write_data+=np.abs(self.input_range[0])
            #double precision is memory consumptive
            write_data=np.asarray(write_data,dtype='float32')
            #add masking information - do this last
            if self.mask is not None: #add a mask channel
                write_data = np.ma.array(write_data, mask = self.mask,dtype='float32')
            if self.output_extension=='png':
                write_data/=np.max(write_data)
                write_data*=255
                f = open(strSavePath,'wb')
                w = png.Writer(*(write_data.shape[1],write_data.shape[0]),greyscale=(self.cmap=='gray'))
                w.write(f,write_data)
                f.close()
            elif self.output_extension=='eps':
                fig = plt.figure()
                ax = plt.Axes(fig,[0,0,1,1])
                ax.set_axis_off()
                fig.add_axes(ax)
                img = ax.imshow(write_data,cmap=self.cmap,vmin=self.input_range[0],vmax=self.input_range[1], interpolation="none")
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                if self.get_val('colorbar',True):
                    cb = fig.colorbar(img)
                    # cb.set_clim(self.input_range[0],self.input_range[1])
                plt.savefig(strSavePath, format="eps",bbox_inches='tight')
            elif self.output_extension=='tif':
                output = tif.open(strSavePath, mode='w')
                #saving as tiff
                if write_data.ndim==3:
                    #axes swapping due to an oddity in the tiff
                    #package, writes 3-D ndarray RowsXColumnsXFrames by default
                    output.write_image(write_data.swapaxes(2,0).swapaxes(2,1))
                elif write_data.ndim==2:    
                    output.write_image(write_data)
                else:
                    raise ValueError('unsupported # of dims in output.write_image tif')
                output.close()
            elif self.output_extension=='csv':
                #2d matrix to a csv file with columns  row0....rowN...col0...colN
                col_iterator = []
                rows = write_data.shape[0]
                cols = write_data.shape[1]
                max_dim = max(rows,cols)
                table=np.zeros([max_dim,rows+cols])
                if self.mask is not None: #add a mask channel
                    write_data = ~write_data.mask * write_data.data
                #do the rows first
                col_ix = 0
                for j in np.arange(rows):
                    table[0:cols,col_ix] = write_data[j,:]
                    col_ix+=1
                    col_iterator.append('row'+str(j))
                for j in np.arange(cols):
                    table[0:rows,col_ix] = write_data[:,j]
                    col_ix+=1
                    col_iterator.append('col'+str(j))              
                #round to save memory
                # table = np.around(table,decimals=4) 
                # pdb.set_trace()
                self.save_csv(strSavePath[0:-4],data_override=table,col_iterator_override=col_iterator,precision=4)
            else:
                raise ValueError('unsupported extension')
        super(OutputImage,self).save()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputImage(ps_parameters,str_section)
