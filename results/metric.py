#!/usr/bin/python -tt
from numpy import arange, array, maximum, zeros
from py_utils.section import Section
import matplotlib
import matplotlib.pyplot as plt
from py_utils.results.defaults import DEFAULT_CSV_EXT,DEFAULT_KEY

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
        #if this metric should only be updated once (such as an initial solution)
        self.update_once = self.get_val('updateonce',True)
        self.update_enabled = True
        self.figure_location = self.get_val('figurelocation',True)
        self.figure = None
        self.figure_number = None
        self.data = []
        self.ylabel = self.get_val('ylabel',False)
        self.key = self.get_val('key',False,DEFAULT_KEY)
        self.title = self.get_val('title',False)
        self.print_enabled = self.get_val('print',True)
        self.plot_enabled = self.get_val('plot',True, True)
        self.output_format = self.get_val('outputformat',False,'csv')
        #2 element vector, beginning and end to crop for plotting
        self.crop_plot = maximum(zeros(2),self.get_val('cropplot',True)) 
        self.crop_plot.dtype='uint8'
        self.has_csv = self.get_val('hascsv',True,True)
        self.save_often = self.get_val('saveoften',True,False)

    def plot(self):
        if self.plot_enabled:
            plt.figure(self.figure_number)
            slice_start = self.crop_plot[0]
            slice_end = self.crop_plot[1]
            if slice_end==0:
                slice_end = None
            sl=slice(slice_start,slice_end)
            plt.plot(array(self.data)[sl])
        
    def update(self, value=None):
        if value != None and self.update_enabled:
            self.data.append(value)
            if self.update_once:
                self.update_enabled=False    
        if self.print_enabled:
            print self.get_val('key',False) + ':\t' + str(self.data[-1])  

    def save(self,strPath='/home/'):
        #empty out the data after the derived class has done the saving
        self.data = []
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Metric(ps_parameters,str_section)
