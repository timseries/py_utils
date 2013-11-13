#!/usr/bin/python -tt
from py_utils import parameter_struct
class Section(object):
    """
    Base class for defining other classes which inherit properties from a config.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Section.
        """       
        self.ps_parameters = ps_parameters
        self.str_section = str_section
        self.str_object_name = ps_parameters.get_section_dict(str_section)['name']
        str_class_name = self.__class__.__name__
        if self.str_object_name!=str_class_name and str_class_name!='Section':
            raise Exception("section class name doesnt match class spec" + \
                            self.str_object_name + '!=' + str_class_name)
        self.dict_section = self.get_section_dict()
        
    def get_section_dict(self):
        return self.ps_parameters.get_section_dict(self.str_section)
    
    def get_subsection_strings(self,str_key):
        ls_subsections = self.dict_section[str_key].split()

    def get_subsections(self,str_key):
        """
        Returns section objects specified from a sections str_key field
        """       
        ls_subsections = self.get_subsection_strings(str_key)
        return [Section(self.ps_parameters,str_section) for str_section in ls_subsections]

    def get_val(self,str_key,lgc_val_numeric=0):
        """
        Returns the value corresponding to str_key in this section, with defaults.
        """       
        if lgc_val_numeric:
            val = 0
        else:
            val = ''
        if self.dict_section.has_key(str_key):
            val = self.dict_section[str_key]
        return val