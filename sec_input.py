#!/usr/bin/python -tt
from libtiff import TIFF as tif
from PIL import Image
from scipy import misc
import numpy as np
import os

from py_utils.section import Section

import pdb

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
        self.filemember = self.get_val('filemember', False) #used for structured file
        self.filename = self.get_val('filename', False)
        #use the config's dir as a reference pt if path not specified or not full
        self.filepath = self.filedir + self.filename
        if not (self.filepath[0] == '/' or self.filepath == ''):
            self.filepath = self.ps_parameters.str_file_dir + '/' + self.filepath
          
    def read(self,dict_in,return_val=False):
        """
        Read a file, branch on the filetype.
        """
        if self.filename=='class_directories':
            #using the directory structure to build a dictionary of lists, each
            #dictionary entry corresponding to a different class, with the 
            #class exemplars elements of the lists
            dict_in['x'] = {}
            file_tuple_numbered = enumerate(os.walk(self.filedir))
            for cls_index,file_tuple in file_tuple_numbered:
                if cls_index==0:
                    entries = file_tuple[1]
                else:                    
                    dict_in['x'][entries[cls_index-1]] = file_tuple[2]
                    dict_in['x'][entries[cls_index-1]].sort()
                    #read in the files
                    for file_index,entry in enumerate(dict_in['x'][entries[cls_index-1]]):
                        filename=dict_in['x'][entries[cls_index-1]][file_index]
                        filepath = self.filedir + '/' + entries[cls_index-1] + '/' + filename
                        dict_in['x'][entries[cls_index-1]][file_index] = self.read_single_file(filepath)
        else: #single file case    
            ary_image = self.read_single_file(self.filepath)
            if return_val:
                return ary_image
            else:
                dict_in['x'] = ary_image

    def read_single_file(self,filename):
        str_extension = filename.split('.')[-1]
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
        elif str_extension == 'jpg':
            ary_image = misc.imread(filename)
        else:
            raise Exception('unsupported format: ' + str_extension)
        return ary_image

    class Factory:
        def create(self,ps_parameters,str_section):
            return Input(ps_parameters,str_section)
