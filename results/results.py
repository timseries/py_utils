#!/usr/bin/python -tt
import numpy as np
from numpy import arange, floor, mod
from numpy import asarray as aa
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf
#matplotlib.use('TkAgg')   #backend setting
import pylab
import matplotlib._pylab_helpers
import matplotlib.pyplot as plt
import Tkinter
import os

class Results(Section):
    """
    Class for defining a collection of metrics to update. Contains methods to 
    determine placement of metric figures on the screen in a tiled environment.
    """
    
    def __init__(self, ps_parameters, str_section):
        """
        Class constructor for Results.
        """       
        super(Results,self).__init__(ps_parameters,str_section)
        self.ls_metric_names = self.get_val('metrics',False)
        if self.ls_metric_names.__class__.__name__ == 'str':
            self.ls_metric_names = [self.ls_metric_names]
        self.ls_metrics = [sf.create_section(ps_parameters, self.ls_metric_names[i]) \
                           for i in arange(len(self.ls_metric_names))]
        self.grid_size = aa([self.get_val('figuregridwidth',True), \
                             self.get_val('figuregridheight',True)], dtype = np.int)
        self.desktop = self.get_val('desktop',True)
        self.row_offset = self.get_val('rowoffset',True)
        self.int_overlap = max(5,self.get_val('overlap',True))
        #get scrren info
        screen = os.popen("xrandr -q -d :0").readlines()[0]
        self.screen_size =  aa([int(screen.split()[7]), \
                                int(screen.split()[9][:-1])], dtype = np.int)
        self.arrange_metric_windows() #figure out the coordinates
        
    def update(self,dict_in):
        for metric in self.ls_metrics:
            metric.update(dict_in)
             
    def arrange_metric_windows(self):
        """
        Determine the grid placement of metric figure windows and assign figure numbers. 
        Should only be called once.
        """       
        figures=[manager.canvas.figure \
                 for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        int_fig_offset = len(figures) - 1
        for metric in self.ls_metrics:
            #make row offset adjustment
            metric.figure_location += self.grid_size[0] * self.row_offset
            metric.w_size = self.screen_size / self.grid_size
            int_row_offset = int((metric.figure_location - .1) / self.grid_size[0])
            int_col_offset = int(mod(metric.figure_location, self.grid_size[0] + 1))
            metric.w_coords = aa([int_col_offset * metric.w_size[0] + \
                                  self.desktop * self.screen_size[0], \
                                  int_row_offset * metric.w_size[1]])
            #creating the metric's figures
            metric.figure_number = int_fig_offset + metric.figure_location
            metric.figure = plt.figure(metric.figure_number)
            wm = plt.get_current_fig_manager()
            wm.window.wm_geometry(str(metric.w_size[0]) + "x" + \
                                  str(metric.w_size[1]) + "+" + \
                                  str(metric.w_coords[0]) + "+" + \
                                  str(metric.w_coords[1])) #does not work with MAC (yet)
                                  
    def clear(self):
        """
        Clear the data out of the metrics 
        """       
        for metric in self.ls_metrics:
            metric.data = []
            
    def close(self):
        plt.close('all')
        
    def display(self):
        """
        Plot the metrics
        """       
        for metric in self.ls_metrics:
            metric.plot()
        plt.show()

    class Factory:
        def create(self,ps_parameters,str_section):
            return Results(ps_parameters,str_section)
        