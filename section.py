#!/usr/bin/python -tt
from py_utils import parameter_struct
from numpy import array
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
            raise Exception("section class name doesnt match object class name " + \
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

    def get_val(self,str_key,lgc_val_numeric=False,default_value=0):
        """
        Returns the value corresponding to a key in this section, with defaults. 

        :param str_key: key to look up in this section
        :param lgc_val_numeric: boolean, output should be numeric (true) or string (false)

        :returns val: Either a string, a numeric value (float or int, depending), \
        or a list of ints.
        """       
        if lgc_val_numeric:
            val = default_value
        else:
            if default_value == 0:
                val = ''
            else:
                val = default_value
        if self.dict_section.has_key(str_key):
            val = self.ps_parameters.config.get(self.str_section,str_key).strip()
            if lgc_val_numeric:
                #test for numeric arrays separated by a space
                if ' ' in val: 
                    try: 
                        val=array([int(val.split()[i]) 
                                   for i in range(len(val.split()))])
                    except ValueError:
                        val=array([float(val.split()[i]) 
                                   for i in range(len(val.split()))])
                else:
                    val1 = self.ps_parameters.config.getfloat(self.str_section,str_key)
                    try:
                        val2 = self.ps_parameters.config.getint(self.str_section,str_key)
                    except ValueError:
                        val2 = val1
                    
                    if val1 == val2:
                        val = val2
                    else:
                        val = val1    
            else:
                if ',' in val:
                    val=val.split(',')
        return val

    class Factory:
        def create(self,ps_parameters,str_section):
            return Section(ps_parameters,str_section)
