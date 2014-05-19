#!/usr/bin/python -tt
from libtiff import TIFF as tif
from PIL import Image
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from collections import OrderedDict
import os
import csv
import cPickle

from py_utils.section import Section

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
        #used for structured file, or specific class directories, or even to specifiy no reading at all 'none'
        self.filemember = self.get_val('filemember', False) 
        self.filename = self.get_val('filename', False)
        #use the config's dir as a reference pt if path not specified or not full
        self.filepath = self.filedir + '/' + self.filename
        if not (self.filepath[0] == '/' or self.filepath == ''): #absolute, or no, path specified
            self.filepath = self.ps_parameters.str_file_dir + '/' + self.filepath
          
    def read(self,dict_in,return_val=False):
        """
        Read a file, branch on the filetype.
        """
        #for the classifcation inputs, dict_in['x'] should have this structure
        #dict_in['x']={'class1':[exemplar1,exemplar2,...],'class2':[exemplar1,exemplar2,...]}
        #where exemplar is a two list ['entryid',data], data is an nparray
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
                    #read in the files instead of having filenames
                    for file_index,entry in enumerate(dict_in['x'][entries[cls_index-1]]):
                        filename=dict_in['x'][entries[cls_index-1]][file_index]
                        filepath = self.filedir + '/' + entries[cls_index-1] + '/' + filename
                        exemplar = [entry,self.read_single_file(filepath)]
                        dict_in['x'][entries[cls_index-1]][file_index] = exemplar
            #now, use the ordereddict to give the dict in dict_in['x'] a key-sorted order
            dict_in['x'] = OrderedDict(sorted(dict_in['x'].items(), key=lambda t: t[0]))
        elif self.filename[:9]=='class_csv':
            #use a csv file to build the dictionary of list, 
            #each entry corresponding to a different class, with the
            #class exemplars being elements of the lists    
            #useful for challenges, when you don't know the test labels
            dict_in['x'] = {}
            with open(self.filedir+'/'+self.filename, 'rb') as csvfile:
                csvreader = csv.reader(csvfile)
                labels=[row[1] for row in csvreader]
                classes=sorted(list(set(labels)))
                for _class in classes:
                    #initialize with None elements to get partitioning right in sec_observe
                    dict_in['x'][_class]=[]
            with open(self.filedir+'/'+self.filename, 'rb') as csvfile:
                csvreader = csv.reader(csvfile) #start from beginning    
                for row in csvreader:
                    file_path=self.filedir+'/'+row[0]+'.jpg'
                    if self.filemember=='ids':
                        exemplar=[row[0],None] #just the id
                    else:
                        exemplar=[row[0],self.read_single_file_file(file_path)]
                    dict_in['x'][row[1]].append(exemplar)
            dict_in['x'] = OrderedDict(sorted(dict_in['x'].items(), key=lambda t: t[0]))         
        else: #single file case    
            file_data = self.read_single_file(self.filepath)
            if return_val:
                return file_data
            else:
                dict_in['x'] = file_data

    def read_single_file(self,filepath):
        str_extension = filepath.split('.')[-1]
        if str_extension == 'tif':#2d image or 3d stack of images
            input_file = tif.open(self.filepath, mode='r')
            volume = list(input_file.iter_images())
            if len(volume) == 1:
                file_data = input_file.read_image()
            else: #we have a volume    
                file_data = np.zeros([volume[0].shape[0],volume[0].shape[1],len(volume)],dtype='float32')
                for index,image in enumerate(volume):
                    file_data[:,:,index] = image
            input_file.close()
        elif str_extension == 'jpg' or str_extension == 'png':
            file_data = misc.imread(filepath)
        elif str_extension == 'pkl':
            filehandler = open(filepath, 'r') 
            file_data = cPickle.load(filehandler)
            filehandler.close()
        elif str_extension == 'json':
            raise ValueError('json not coded yet')
        else:
            raise Exception('unsupported format: ' + str_extension)
        return file_data

    class Factory:
        def create(self,ps_parameters,str_section):
            return Input(ps_parameters,str_section)
