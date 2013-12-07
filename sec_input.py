#!/usr/bin/python -tt
from py_utils.section import Section
from libtiff import TIFF as tif
import Image

class Input(Section):
    """
    Input class for handling reading of any type of data file for a solver pipeline.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Input.
        """       
        super(Input,self).__init__(ps_parameters,str_section)
        self.filedir = self.get_val('filedir', False)
        self.filemember = self.get_val('filemember', False) #used for stuctured file
        self.filename = self.get_val('filename', False)
        #use the config's dir as a reference pt if path not specified or not full
        self.filepath = self.filedir + self.filename
        if not (self.filepath[0] == '/' or self.filepath == ''):
            self.filepath = self.ps_parameters.str_file_dir + '/' + self.filepath
          
    def read(self,dict_in):
        """
        Read a file, branch on the filetype.
        """
        str_extension = self.filename.split('.')[-1]
        if str_extension == 'tif':
            input_file = tif.open(self.filepath, mode='r')
            ary_image = input_file.read_image()
            input_file.close()
        else:
            raise Exception('unsupported format: ' + str_extension)
        
        dict_in['x'] = ary_image
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Input(ps_parameters,str_section)
