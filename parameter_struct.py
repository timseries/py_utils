#!/usr/bin/python -tt
import ConfigParser
import csv
import os
import os.path
from os.path import exists, dirname, expanduser
import itertools

import pdb

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
        if not exists(str_file_path):
            raise Exception("file " + str_file_path + " nonexistent")    
        self.config.read(str_file_path)
        if self.config.sections() == []:
            raise Exception("file " + str_file_path + " invalid config")    
        self.str_file_path = str_file_path
        pathsplit=os.path.split(str_file_path)
        self.str_file_dir = pathsplit[0]
        self.str_fname = pathsplit[1]
        self.str_fname_noext = self.str_fname.split('.')[0]
        self.str_ext = self.str_fname.split('.')[1]
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

    def generate_configs(self):
        """For any values which contain a sweep() directive, 
        the arguments are parsed and a new configuration file is generated, 
        this configuration file's prefix. All possible combinations.
        Write each permutation of sweepable parameters to a separate config
        file and returns the list of filenames
        """
        #
        #create a dictionary of dictionaries
        #Keys: section names
        #Values: dicts
                   #Keys: param_names
                   #values: param_values
                   
        sweepables = []           
        config_paths = []
        for section_name in self.section_names:
            for param_val_pairs in self.config.items(section_name):
                if 'sweep(' in param_val_pairs[1]:
                    sweepable = param_val_pairs[1][6:-1].split(',')
                    sweepables.append([section_name + '-' + param_val_pairs[0] + '-' +  item for item in sweepable])
                    
        sweepable_product =  itertools.product(*sweepables)
        for sweepable_product_item in sweepable_product:
            itemstring = ''
            for entry in sweepable_product_item:
                section, param, paramval = entry.split('-')
                self.set_key_val_pairs(section, [param], [paramval])
                itemstring += param + ':' + paramval + '-'
            itemstring = itemstring[:-1]
            config_path = self.str_file_dir + '/' + self.str_fname_noext + '-' + itemstring + '.' + self.str_ext
            self.write(config_path)
            config_paths.append(config_path)
        return config_paths

    def get_section_dict(self,str_section):
        return self.config._sections[str_section]
