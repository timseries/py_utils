#!/usr/bin/python -tt
import numpy as np
from py_utils.results.metric import Metric
import os
from py_utils.results.defaults import DEFAULT_SLICE,DEFAULT_IMAGE_EXT

class OutputOneDim(Metric):
    """
    OutputOneDim metric class, for storing a single number or label vs iteration or sample index.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(OutputOneDim,self).__init__(ps_parameters,str_section)        
        self.slices = None
        self.slice = self.get_val('slice',True, DEFAULT_SLICE)
        self.last_frame_only = self.get_val('lastframeonly',True) 
        
    def update(self,dict_in):
        """
        Store the 1D signal along the rows.
        """
        value=dict_in[self.key]
        if self.data == []:
            #there always needs to be a dict_in['x'] reference image
            # to compute the input range, for scaling the output range
            self.slices = [slice(0,None,None) if i < 2 else 
                           slice(max(0,min(self.slice,value.shape[i])),None,None) 
                           for i in xrange(value.ndim)]
        super(OutputOneDim,self).update(value[self.slices])    
    def save(self,strPath='/home/outputimage/'):
        #TODO: superclass this and outut_image to commonalize this code
        if len(self.data)==0:
            return
        if self.last_frame_only or self.update_once:
            frame_iterator=[('',self.data[-1])]
            ix_offset=''
        else:
            frame_iterator=enumerate(self.data)
            files_enumerated = enumerate(os.walk(os.path.dirname(strPath)))
            base_name=os.path.basename(strPath)
            files,dir_info=files_enumerated.next()
            if self.save_often:
                ix_offset=len([file_name for file_name in dir_info[2] if base_name in file_name])
            else:    
                ix_offset=0
        for ix,frame in frame_iterator:
            strSavePath = strPath + str(ix_offset+ix) #save_csv() adds the extension
            write_data = frame[self.slices]
            write_data=np.asarray(write_data,dtype='float32')
            #format the frame to use the super's Metric.save_csv (use 2d array with one column)
            write_data_temp = np.zeros((write_data.size, 1))
            write_data_temp[:,0] = write_data
            write_data = write_data_temp
            super(OutputOneDim, self).save_csv(strPath=strSavePath,data_override = write_data)
        super(OutputOneDim,self).save()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputOneDim(ps_parameters,str_section)
