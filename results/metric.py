#!/usr/bin/python -tt
from numpy import arange, array, maximum, zeros
from py_utils.section import Section
import matplotlib
import matplotlib.pyplot as plt
from py_utils.results.defaults import DEFAULT_CSV_EXT

class Metric(Section):
    """
    Base class for defining a metric. See :func:`py_utils.section.Section.__init__`.
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
        self.key = self.get_val('key',False)
        self.title = self.get_val('title',False)
        self.print_values = self.get_val('print',True)
        self.output_format = self.get_val('outputformat',False,'csv')
        self.crop_plot = maximum(zeros(2),self.get_val('cropplot',True)) #2 element vector, beginning and end to crop for plotting
        self.has_csv = self.get_val('hascsv',True,True)

    def plot(self):
        plt.figure(self.figure_number)
        
        slice_start = self.crop_plot[0]
        slice_end = self.crop_plot[1]
        if slice_end==0:
            slice_end = None
        sl=slice(slice_start,slice_end)
        plt.plot(array(self.data)[sl])

    def update(self, value=None):
        if value != None:
            self.data.append(value)
        if self.print_values:
            print self.get_val('key',False) + ':\t' + str(self.data[-1])  

    def save(self,strPath='/home/'): pass
         
    class Factory:
        def create(self,ps_parameters,str_section):
            return Metric(ps_parameters,str_section)
