#!/usr/bin/python -tt
import numpy as np
from numpy import arange, floor, mod
from numpy import asarray as aa
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf
from py_utils.results.defaults import DEFAULT_CSV_EXT,DEFAULT_PARAMETERS_FILE,DEFAULT_ZEROS

#matplotlib.use('TkAgg')   #backend setting
import pylab
import matplotlib._pylab_helpers
import matplotlib.pyplot as plt
import Tkinter
import os
import time
import re
import datetime
import csv 

import pdb

class Results(Section):
    """
    Class for defining a collection of metrics to update. Contains methods to 
    determine placement of metric figures on the screen in a tiled environment.
    """
    
    def __init__(self, ps_parameters, str_section):
        """Class constructor for Results.
        """       
        super(Results,self).__init__(ps_parameters,str_section)
        self.ls_metric_names = self.get_val('metrics',False)
        if self.ls_metric_names.__class__.__name__ == 'str':
            self.ls_metric_names = [self.ls_metric_names]
        self.ls_metrics = [sf.create_section(ps_parameters, self.ls_metric_names[i]) \
                           for i in arange(len(self.ls_metric_names))]
        self.ls_metrics_csv = [self.ls_metrics[i] 
                               for i in arange(len(self.ls_metrics))
                               if self.ls_metrics[i].has_csv]                   
        self.ls_metrics_no_csv = [self.ls_metrics[i] 
                                  for i in arange(len(self.ls_metrics))
                                  if not self.ls_metrics[i].has_csv]                  
        self.grid_size = aa([self.get_val('figuregridwidth',True), \
                             self.get_val('figuregridheight',True)], dtype = np.int)
        self.desktop = self.get_val('desktop',True)
        self.row_offset = self.get_val('rowoffset',True)
        self.int_overlap = max(5,self.get_val('overlap',True))
        self.save_interval = self.get_val('saveinterval',True)
        self.output_directory = self.get_val('outputdirectory',False)
        #default to the configuration file's filename as a prefix
        self.output_filename = self.get_val('outputfilename',False,
                                            default_value=self.get_params_fname(False))
        self.overwrite_results = self.get_val('overwriteresults',True)
        self.zeros = self.get_val('zeros', True, DEFAULT_ZEROS)
        self.display_enabled = not self.get_val('disablefigures',True)
        self.monitor_count=0
        self.resolutions=tuple()
        
        #get screen info
        if self.display_enabled:
            #try to find a display
            screen = os.popen("xrandr -q -d :0").readlines()
            if len(screen)>0:
                ls_res=re.findall(' connected [0-9]*x[0-9]*', ''.join(screen))
                ls_res=re.findall('[0-9]*x[0-9]*', ''.join(ls_res))
                #the ordering of these assumes the screens are in the same order
                #if this is not the case, simply rearrange your screens ;-)
                self.resolutions=tuple([np.array(res.split('x'),'uint16') for res in ls_res])
                self.monitor_count = len(self.resolutions)
                self.desktop=np.mod(self.desktop,self.monitor_count)
                self.arrange_metric_windows() #figure out the coordinates
            else: #turn display off    
                self.monitor_count=0
                self.display_enabled = False
        #multiple viewport/desktop support not enabled, so wrapping the desktop to the range
        #of available monitors
        #create a folder in the output directory with the current minute's time stamp
        if self.output_directory=='':
            print ('Not writing results to file no output dir specified')
            return None
        st = '/' + self.output_filename
        self.output_directory += st + '/'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        if not self.overwrite_results:
            #old timestamping method, keep in case this is deemed better in the future
            # st += '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
            #get the count of the current numbered directories
            files_enumerated = enumerate(os.walk(self.output_directory))
            files,dir_info=files_enumerated.next()
            results_count_string=str(len(dir_info[1])).zfill(self.zeros)
            self.output_directory += '/' + results_count_string + '/'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        #save the parameters to this folder as ini, and write as csv
        self.ps_parameters.write(self.output_directory + self.output_filename + '_config.ini')
        self.ps_parameters.write_csv(self.output_directory + self.output_filename + '_config.' + DEFAULT_CSV_EXT)

    def update(self,dict_in):
        """Update the metrics in this results collection.
        """
        for metric in self.ls_metrics:
            metric.update(dict_in)
            if metric.save_often:
                metric.save(self.get_metric_path(metric))
    def save(self): #really should be save_metrics
        """Save the metrics in this results collection to file. 
        This aggregates all fo the 'csv' output metrics together into one csv pdatefile.
ii        The other metrics are dealt with separately.
        Does not overwrite results by default, and instead creates a new time-stamped subdirectory of self.output_directory
        """
        #collect all of the metrics into a table (list of lists, one list per row)
        #these metrics can be written to a csv file
        if len(self.ls_metrics_csv)>0:
            int_rows = max([len(metric.data) for metric in self.ls_metrics_csv])
            table = [[j] + [self.csv_cell(metric.data[j]) if j < len(metric.data) 
                            else self.csv_cell(metric.data[-1]) 
                            for metric in self.ls_metrics_csv] 
                            for j in xrange(int_rows)]
            #start a new csv file, and save the csv metrics there
            headers = [metric.key for metric in self.ls_metrics_csv] 
            headers.insert(0,'n')
            with open(self.output_directory + self.output_filename + '_metrics.' + DEFAULT_CSV_EXT, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(table)
        #save the other metrics to file by invoking their respective save methods
        for metric in self.ls_metrics_no_csv:
            self.save_metric(metric)

    def save_metric(self,metric):
        metric.save(self.get_metric_path(metric))

    def get_metric_path(self,metric):
        return self.output_directory + self.output_filename + '_' + metric.key

    def csv_cell(self,metric_datum):
        '''convert metric_datum into an acceptable format for a csv cell
        tuples-> string of ;-delimited values
        '''
        return metric_datum
    
    def get_output_path(self):
        return self.output_directory
        
    def arrange_metric_windows(self):
        """
        Determine the grid placement of metric figure windows and assign figure numbers. 
        Should only be called once during the object's creation.
        """       
        #see if there are any other figures open so as not to overwrite their
        #contents
        figures=[manager.canvas.figure \
                 for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        int_fig_offset = max(0,len(figures) - 1)
        for metric in self.ls_metrics:
            #make row offset adjustment
            metric.figure_location += self.grid_size[0] * self.row_offset
            metric.w_size = self.resolutions[self.desktop] / self.grid_size
            int_row_offset = int((metric.figure_location + .1) / self.grid_size[0])
            int_col_offset = int(mod(metric.figure_location, self.grid_size[0]))
            metric.w_coords = aa([int_col_offset * metric.w_size[0] + \
                                  self.desktop * self.resolutions[self.desktop][0], \
                                  int_row_offset * metric.w_size[1]])
            #creating the Metrics' figure windows
            metric.figure_number = int_fig_offset + metric.figure_location
            metric.figure = plt.figure(metric.figure_number)
            wm = plt.get_current_fig_manager()
            wm.window.wm_geometry(str(metric.w_size[0]) + "x" + \
                                  str(metric.w_size[1]) + "+" + \
                                  str(metric.w_coords[0]) + "+" + \
                                  str(metric.w_coords[1]))
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
        if self.display_enabled:
            for metric in self.ls_metrics:
                metric.plot()
            plt.show()
        else:
            print 'no display enabled'

    class Factory:
        def create(self,ps_parameters,str_section):
            return Results(ps_parameters,str_section)
        
