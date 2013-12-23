#!/usr/bin/python -tt
from numpy import arange, array
from py_utils.section import Section
import matplotlib
import matplotlib.pyplot as plt
class Metric(Section):
    """
    Base class for defining a metric
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Metric.
        """       
        super(Metric,self).__init__(ps_parameters,str_section)
        self.w_coords = None #list, the x, y coords of this metric
        self.w_size = None
        self.figure_location = self.get_val('figurelocation',True)
        self.figure = None
        self.figure_number = None
        self.data = []
        self.ylabel = self.get_val('ylabel',False)
        self.title = self.get_val('title',False)
        self.print_values = self.get_val('print',True)
        
    def plot(self):
        plt.figure(self.figure_number)
        plt.plot(array(self.data))

    def update(self):
        if self.print_values:
            print self.get_val('key',False) + ':\t' + str(self.data[-1])  
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Metric(ps_parameters,str_section)
