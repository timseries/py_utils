#!/usr/bin/python -tt
import numpy as np
from numpy import arange, array, maximum, zeros
from py_utils.section import Section
import matplotlib
import itertools
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpldatacursor import datacursor
import matplotlib.cm as cm
from py_utils.results.defaults import DEFAULT_CSV_EXT, DEFAULT_KEY, DEFAULT_EXT

import csv 

import pdb

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
        self.output_extension = self.get_val('outputextension', False, DEFAULT_EXT)
        #2 element vector, beginning and end to crop for plotting
        self.crop_plot = maximum(zeros(2),self.get_val('cropplot',True)) 
        self.crop_plot.dtype='uint8'
        self.has_csv = self.get_val('hascsv',True,True)
        self.save_often = self.get_val('saveoften',True,False)
        self.legend_labels = None
        self.logy = self.get_val('logy',True)
        self.logx = self.get_val('logx',True)
        if self.logy and self.logx:
            self.plotfun = plt.loglog
        elif self.logy:
            self.plotfun = plt.semilogy
        elif self.logx:
            self.plotfun = plt.semilogx
        else:
            self.plotfun = plt.plot
        self.ylim = self.get_val('ylim',True,None,False)    
        self.xlim = self.get_val('xlim',True,None,False)    
        self.ylabel = self.get_val('ylabel',False,None,False)    
        self.xlabel = self.get_val('xlabel',False,None,False)    
        self.legend_pos = None
        if self.xlabel is None:
            self.xlabel = r'Iteration'
        self.legend_cols = 1
        self.save_key = None
        self.last_frame_only = False
            
    def plot(self):
        if self.plot_enabled:
            plt.figure(self.figure_number)
            slice_start = self.crop_plot[0]
            slice_end = self.crop_plot[1]
            if slice_end==0:
                slice_end = None
            sl=slice(slice_start,slice_end)
            datacursor(display='single')
            if self.data[0].__class__.__name__=='ndarray':
                # colors = cm.rainbow(np.linspace(0, 1, len(array(self.data)[0])))
                linestyles = itertools.cycle(['-','--','-.',':'])
                colors = itertools.cycle(["r","g","b","c",'m','k','orange'])
                num_series = len(self.data[0])
                series_y_all = array(self.data).transpose()
            else:
                linestyles = itertools.cycle(['-'])
                colors = itertools.cycle(['black'])
                num_series = 1    
                series_y_all = [array(self.data)]
            series_x = np.arange(len(series_y_all[0])) 
            for series_y in series_y_all:
                self.plotfun(series_x,series_y,color=next(colors),linestyle=next(linestyles))
                
            if self.ylim is not None:    
                plt.ylim(self.ylim)
            if self.xlim is not None:    
                plt.xlim(self.xlim)
            else:
                plt.xlim([series_x[0],series_x[-1]])    
            if self.xlabel is not None:
                plt.xlabel(self.xlabel)
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.legend_pos is None:
                self.legend_pos = 'upper center'
            if self.legend_labels is not None:
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                fontP = FontProperties()
                fontP.set_size('small')
                legend = plt.legend(self.legend_labels,
                                    loc=self.legend_pos,
                                    # bbox_to_anchor = (1.01, 1.01),
                                    ncol = self.legend_cols,
                                    title = self.title,
                                    prop=fontP,labelspacing = .5)
                plt.setp(legend.get_title(),fontsize='x-small')    
        
    def update(self, value=None):
        if value is not None and self.update_enabled:
            if self.last_frame_only:
                self.data = [value]
            else:
                self.data.append(value)
            if self.update_once:
                self.update_enabled=False    
        if self.print_enabled:
            print self.get_val('key',False) + ':\t' + str(self.data[-1])  

    def save(self,strPath='/home/'):
        #empty out the data after the derived class has done the saving
        self.data = []

    def save_csv(self,strPath='/home/outputimage/',data_override = None,col_iterator_override = None, precision=None):
        '''When the iteration data is saved as a multi-series/array, need to parse and create new columns
        col_iterator_override can be any iterable to provide column names, otherwise column indices are used
        '''
        if data_override is not None:
            data = data_override
        else:
            data = self.data    
        if len(self.data)==0:
            return
        int_rows = len(data)
        col_iterator = xrange(len(data[0]))
        if col_iterator_override is not None:
            col_iterator = col_iterator_override
        table = [[j] + [data[j][k] for k,name in enumerate(col_iterator)] for j in xrange(int_rows)]
            #start a new csv file, and save the csv metrics there
        headers = [self.key + str(k) for k in col_iterator]
        headers.insert(0,'n')
        with open(strPath + '.' + DEFAULT_CSV_EXT, 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            writer.writerow(headers)
            if precision is None:
                writer.writerows(table)
            else:
                fmt_string = '{:1.'+str(int(precision))+'}'
                for row in table:
                    # writer.writerow(['{:3.4e}'.format(x) for x in i])
                    writer.writerow(['{:1.4}'.format(el) if ix >0 else el for ix,el in enumerate(row)])
    class Factory:
        def create(self,ps_parameters,str_section):
            return Metric(ps_parameters,str_section)
