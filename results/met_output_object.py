#!/usr/bin/python -tt
import pickle

from py_utils.results.metric import Metric
from py_utils.results.defaults import DEFAULT_OBJECT_EXT

class OutputObject(Metric):
    """
    Class for outputting an values from a dict (as objects) using json or (c)pickle. Used for storing model 
    parameters, numpy arrays, lists or other intermediate data for efficient future retrieval.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for OutputObject. See :func:`py_utils.results.metric.Metric.__init__`.
        :param ps_parameters: :class:`py_utils.parameter_struct.ParameterStruct` object which should have 'output_extension' and 'keys' in the :param:`str_section` heading.
        
        """
        super(OutputObject,self).__init__(ps_parameters,str_section)
        #build the operator objects            
        self.output_extension = self.get_val('outputextension', False, DEFAULT_OBJECT_EXT)
        self.print_values = 0 #we never want to print array data...
        self.has_csv = False #we can't save these to csv format like other metrics
        self.dict_in = None
        
    def update(self,dict_in):
        self.dict_in = dict_in
        super(OutputObject,self).update()

    def plot(self): pass #nothing to plot...

    def save(self,strPath='~/outputimage'):
        if self.dict_in == None:
            ValueError('uninitialized dict in outputobject')
        data = self.dict_in[self.key]    
        strPath = strPath + '.' + self.output_extension
        filehandler = open(strPath, 'wb')
        if self.output_extension=='pkl':
            pickle.dump(data, filehandler) 
        elif self.output_extension=='json':
            ValueError('json not available yet')            
        else:
            ValueError('unsupported extension')
        filehandler.close()
    class Factory:
        def create(self,ps_parameters,str_section):
            return OutputObject(ps_parameters,str_section)
