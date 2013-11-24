#!/usr/bin/python -tt
from numpy import arange
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
        
    def plot(self):
        plt.figure(self.figure_number)
        plt.plot(np.array(self.data))

    class Factory:
        def create(self,ps_parameters,str_section):
            return Metric(ps_parameters,str_section)
