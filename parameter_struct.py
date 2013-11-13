#!/usr/bin/python -tt
import ConfigParser
class ParameterStruct(object):
    """
    Base class for defining other classes which inherit properties from a config.
    """
    
    def __init__(self,str_file_path):
        """
        Class constructor for Section.
        """       
        self.config = ConfigParser.ConfigParser(allow_no_value=True)
        self.config.read(str_file_path)
        self.str_file_path = str_file_path
        
    def write(self,str_file_path=None):
        if str_file_path == None:
            str_file_path = self.str_file_path
        file = open(str_file_path)
        self.config.write(file)
        file.close()

    def set_key_val_pairs(self,str_section,ls_keys,ls_values):
        if len(ls_keys)!=len(ls_values) or str_section=='':
            raise Exception("key/val spec not same lengths, or invalid section")
        else:
            for key, value in zip(ls_keys,ls_values):
                self.config.set(str_section, key, value)
                
    def get_section_dict(self,str_section):
        return self.config._sections[str_section]