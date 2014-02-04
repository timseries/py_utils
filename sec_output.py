#!/usr/bin/python -tt
from py_utils.section import Section
from libtiff import TIFF as tif
import numpy as np
import Image

class Output(Section):
    """
    Output class for writing solution/data to file.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Output.
        """       
        super(Output,self).__init__(ps_parameters,str_section)
        self.filedir = self.get_val('filedir', False)
        self.filemember = self.get_val('filemember', False) #used for stuctured file
        self.filename = self.get_val('filename', False)
        self.filepath = self.filedir + self.filename
        if not (self.filepath[0] == '/' or self.filepath == ''):
            self.filepath = self.ps_parameters.str_file_dir + '/' + self.filepath
          
    def write(self,dict_in,return_val=False):
        """
        Read a file, branch on the filetype.
        """
        str_extension = self.filename.split('.')[-1]
        if str_extension == 'tif':
            input_file = tif.open(self.filepath, mode='r')
            volume = list(input_file.iter_images())
            if len(volume) == 1:
                ary_image = input_file.read_image()
            else: #we have a volume    
                ary_image = np.zeros([volume[0].shape[0],volume[0].shape[1],len(volume)])
                for index,image in enumerate(volume):
                    ary_image[:,:,index] = image
            input_file.close()
        else:
            raise Exception('unsupported format: ' + str_extension)
        if return_val:
            return ary_image
        else:
            dict_in['x'] = ary_image

    def write(self,ary_,return_val=False):        
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Output(ps_parameters,str_section)
