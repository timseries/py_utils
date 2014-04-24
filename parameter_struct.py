#!/usr/bin/python -tt
import ConfigParser
import csv
import os
from os.path import exists, dirname, expanduser

class ParameterStruct(object):
    """
    Base class for defining other classes which inherit properties from a config.
    """
    
    def __init__(self,str_file_path):
        """
        Class constructor for Section.
        """       
        self.config = ConfigParser.ConfigParser(allow_no_value=True)
        str_file_path = expanduser(str_file_path)
        self.config.read(str_file_path)
        if self.config.sections() == []:
            raise Exception("file " + str_file_path + " empty or non-existent")    
        self.str_file_path = str_file_path
        self.str_file_dir = os.path.dirname(os.path.realpath(str_file_path))
        self.section_names = self.config._sections.keys()
        
    def write(self,str_file_path=None):
        if str_file_path == None:
            str_file_path = self.str_file_path
        file = open(str_file_path,'w')
        self.config.write(file)
        file.close()

    def write_csv(self,str_file_path=None):
        if str_file_path == None or not exists(dirname(str_file_path)):
            raise ValueError('need a valid csv filepath')
            #build the headers
        else:    
            headers = [[self.section_names[i] + ":" + \
                        self.get_section_dict(self.section_names[i]).items()[j][0] \
                        for j in xrange(len(self.get_section_dict(self.section_names[i]).items()))] \
                        for i in xrange(len(self.section_names))]
            fullheaders = [headers[i][j] for i in xrange(len(headers)) for j in xrange(len(headers[i]))]
            values = [self.get_section_dict(fullheaders[i].split(':')[0])[fullheaders[i].split(':')[1]] for i in xrange(len(fullheaders))]
        with open(str_file_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fullheaders)
            writer.writerow(values)
            
    def set_key_val_pairs(self,str_section,ls_keys,ls_values):
        if len(ls_keys)!=len(ls_values) or str_section=='':
            raise Exception("key/val spec not same lengths, or invalid section")
        else:
            for key, value in zip(ls_keys,ls_values):
                if value.__class__.__name__!='str':
                    if value.__class__.__name__=='ndarray':
                        value = str(value)
                        value=value[2:-1]#strip off the brackets
                    else:
                        value = str(value)
                self.config.set(str_section, key, value)

    def get_section_dict(self,str_section):
        return self.config._sections[str_section]
