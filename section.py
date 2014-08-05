#!/usr/bin/python -tt
from numpy import array
import inspect
from os.path import expanduser

from py_utils import parameter_struct

import pdb

class Section(object):
    """Base class which inherit parameters from a (*.ini configuration based parameterization).

    Attributes:
        ps_parameters (ParameterStruct): The contains (at least) the parameterization
            for this instance.
        str_section (str): The section within the ps_parameters which defines the
            parameterization for this instance.
        str_object_name (str): The class of the object which this instance parameterizes.
        disct_section (dict): The dict corresponding to this instance.

    """
    
    def __init__(self,ps_parameters,str_section):
        """Class constructor for Section.

        Args
            ps_parameters (ParameterStruct): See ps_parameters attribute.
            str_section (str): See str_section attribute.
        
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

    def get_params(self):
        return self.ps_parameters

    def get_params_fname(self,ext=True):
        if ext==False:
            return self.ps_parameters.str_fname_noext
        return self.ps_parameters.str_fname

    def get_subsections(self,str_key):
        """
        Returns section objects specified from a sections str_key field
        """       
        ls_subsections = self.get_subsection_strings(str_key)
        return [Section(self.ps_parameters,str_section) for str_section in ls_subsections]

    def get_val(self,str_key,lgc_val_numeric=False,default_value=0,rtn_list=False):
        """
        Returns the value corresponding to a key in this section, with defaults. 

        :param str_key: key to look up in this section
        :param lgc_val_numeric: boolean, output should be numeric (true) or string (false)

        :returns val: Either a string, a numeric value (float or int, depending), \
        or a list of ints.
        """       
        #set the default value pass-throughs
        if lgc_val_numeric:
            val = default_value
        else:
            if default_value == 0:
                val = ''
            else:
                val = default_value
        if self.dict_section.has_key(str_key):
            val = self.ps_parameters.config.get(self.str_section,str_key).strip()
            lgc_check_numeric = self.is_number(val)
            if lgc_val_numeric:
                #test for numeric arrays separated by a space first
                if ' ' in val: 
                    try: 
                        val=array([int(val.split()[i]) 
                                   for i in range(len(val.split()))])
                    except ValueError:
                        val=array([float(val.split()[i]) 
                                   for i in range(len(val.split()))])
                else:
                    #try to parse a scalar float or int
                    if lgc_check_numeric:
                        #get the appropriate numeric type, float or int
                        val1 = self.ps_parameters.config.getfloat(self.str_section,str_key)
                        try:
                            val2 = self.ps_parameters.config.getint(self.str_section,str_key)
                        except ValueError:
                            val2 = val1
                        if val1 == val2:
                            val = val2
                        else:
                            val = val1    
                    else:#must have mis-specified lgc_val_numeric, return a string
                        val = val
            else:
                #do some parsing for special strings, such as  
                if ',' in val: #comma-delimited lists or
                    val=val.split(',')
                elif '~' in val: #paths or ...
                    val=expanduser(val)    
        if rtn_list and val.__class__.__name__ != 'list':
            val=[val]            
        return val

    def get_keyword_arguments(self,kwprefix):
        """get all keyword arguments start with kwprefix, build the corresponding
        dictionary and return
        
        """
        kwargs = {}
        len_kwprefix = len(kwprefix)
        orig_keys = self.dict_section.keys()
        keys = [key[len_kwprefix:] for key in orig_keys if kwprefix in key]
        orig_keys = [key for key in orig_keys if kwprefix in key]
        for orig_key,key in zip(orig_keys,keys): #assume vals are numeric, and let the get_val method handle exceptions
            kwargs[key]=self.get_val(orig_key,True)
        return kwargs    
    
    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False        
        
    def getRequiredArgs(self,func):
        args, varargs, varkw, defaults = inspect.getargspec(func)
        if defaults:
            args = args[1:-len(defaults)]
        return args   # *args and **kwargs are not required, so ignore them.

    def missingArgs(self,func, argdict):
        return set(self.getRequiredArgs(func)).difference(argdict)

    def invalidArgs(self,func, argdict):
        args, varargs, varkw, defaults = inspect.getargspec(func)
        if varkw: return set()  # All accepted
        return set(argdict) - set(args)

    def isCallableWithArgs(self,func, argdict):
        misargs = self.missingArgs(func, argdict)
        invalidargs = self.invalidArgs(func, argdict)
        lgc_pass =  not misargs and not invalidargs
        if not lgc_pass:
            if misargs:
                raise Warning(self.str_object_name + ' missing arguments ' + str(misargs))
            if invalidargs:
                raise Warning(self.str_object_name + ' invalid arguments ' + str(invalidargs))
        return lgc_pass    

    class Factory:
        def create(self,ps_parameters,str_section):
            return Section(ps_parameters,str_section)
