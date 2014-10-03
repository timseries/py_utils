#!/usr/bin/python -tt
import numpy as np
from libtiff import TIFF as tif
import matplotlib.pyplot as plt
import png
from PIL import Image
from mpldatacursor import datacursor
from matplotlib.colors import LogNorm 
import os
import copy
from matplotlib.ticker import LogLocator, FormatStrFormatter

from py_utils.results.metric import Metric
from py_utils.results.defaults import DEFAULT_SLICE,DEFAULT_IMAGE_EXT

import pdb

class OrderStatistics(Metric):
    """
    Class for outputting and image or volume, and allowing a redraw/update.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for OrderStatistics. See :func:`py_utils.results.metric.Metric.__init__`.

        :param ps_parameters: :class:`py_utils.parameter_struct.ParameterStruct` object which should have 'slice', 'outputformat', and 'outputdirectory' in the :param:`str_section` heading.
        
        """
        super(OrderStatistics,self).__init__(ps_parameters,str_section)
        #use self.slice=-1 for 'all',otherwise specify a slice number
        self.cmap = self.get_val('colormap',False,'jet')
        self.cmap = copy.copy(plt.cm.get_cmap(self.cmap)) #arbitrary cmap
        self.subsamplefactor = self.get_val('subsamplefactor',True) #how to subsample the histograms
        self.overplotkey = self.get_val('overplotkey',True)
        self.has_csv = False #we can't save these to csv format like other metrics
        self.colorbarlabel = self.get_val('colorbarlabel',True)
        self.overplotvals = []
        self.cbar_range = self.get_val('cbarrange',True)

            
    def update(self,dict_in):
        """Takes a 2D or 3D image or volume . If a volume, display the :attr:`self.slice` if specified, otherwise display volume using Myavi. Aggregate images/slices into a volume to view the reconstruction later, or save volumes
        :param dict_in: Input dictionary which contains the referenece to the image/volume data to display and record. 
        """
        value=np.asarray(sorted(dict_in[self.key][0].energy().flatten(),reverse=True)[0::self.subsamplefactor])
        if self.data == []:
            self.overplotvals=dict_in[self.overplotkey]
        super(OrderStatistics,self).update(value)
    def plot(self,showplot=True):
        if showplot:
            fig = plt.figure(self.figure_number)        
        else:
            fig = plt.figure()        
        ax = plt.Axes(fig,[0,0,1,1])
        #arange the columns of these ordered magnitudes into an image (x axis is number of iterations)
        Z = np.zeros((len(self.data[0]),len(self.data)))
        for ix,el in enumerate(self.data):
            Z[:,ix] = self.data[ix]    
        dmn = np.min(Z)
        dmx = np.max(Z)
        if self.cbar_range.__class__.__name__ != 'ndarray':
            clevs = np.linspace(dmn, dmx, 100)
        else:
            clevs = [10**self.cbar_range[0], 10**self.cbar_range[1]]
        # majorLocator   = plt.LogLocator(5)
        # majorFormatter = plt.FormatStrFormatter('%d')
        im = plt.imshow(Z, cmap=self.cmap, interpolation = 'bilinear', 
                        norm=LogNorm(vmin=clevs[0], vmax=clevs[-1]), aspect= 'auto')#extent=[0, 1, 0, 1])
        plt.rc('text', usetex=True)
        plt.xticks(np.arange(0,len(self.data),10))
        # labels = np.asfarray(sorted([item.get_text() for item in ax.get_yticklabels()]))
        # ax.yaxis.set_major_locator(majorLocator)
        # ax.yaxis.set_major_formatter(majorFormatter)
        # ax.set_yticklabels([str(el/np.max(labels))])
        spacing = 10.0
        yticks = np.arange(0,1,1/spacing)*len(self.data[-1])
        plt.yticks(yticks)
        ax = plt.gca()
        ax.set_yticklabels(sorted([str(int(spacing+100*el/len(self.data[-1]))) for el in yticks],key=lambda t: -int(t)))
        #plotting the superimposed plot
        if self.overplotkey != '':
            xs = np.arange(0,len(self.data))
            #find the percentiles of these overplot vals
            self.overplotvals = np.asarray(self.overplotvals)
            overplot_locs = []
            for ix in xrange(len(self.data)):
                
                margin = 10*self.overplotvals[ix]
                overplot_locs_temp = (np.nonzero((Z[:,ix] + margin > self.overplotvals[ix]) * 
                                            (Z[:,ix] - margin < self.overplotvals[ix]))[0])
                
                overplot_locs.append(overplot_locs_temp[len(overplot_locs_temp)/2])
            
            ys = np.asarray(overplot_locs)
            # ys = len(self.data[0]) - ys
            stop_position=np.nonzero(self.overplotvals[:-1]==self.overplotvals[1:])[0][0]
        if stop_position == 0:
            plt.plot(xs,ys,'k-')
        else:
            plt.plot(xs[:stop_position],ys[:stop_position],'ko-')
            plt.plot(xs[stop_position-1:],ys[stop_position-1:],'k-')
        # plt.plot(np.arange(0,len(self.data)),np.asarray(self.overplotvals))
        if self.ylabel!='':
            plt.ylabel(self.ylabel,fontsize=14)
        if self.xlabel!='':
            plt.xlabel(self.xlabel,fontsize=14)
        if self.get_val('colorbar',True):
            cb = plt.colorbar()
            cb.set_label(self.colorbarlabel,fontsize=14)

    def save(self,strPath='/home/outputimage/'):
        if self.output_extension=='eps':
            self.plot(False)
            strSavePath = strPath + '.' + self.output_extension
            plt.savefig(strSavePath, format="eps",bbox_inches='tight')
        else:
            raise ValueError('unsupported extension')
        super(OrderStatistics,self).save()
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return OrderStatistics(ps_parameters,str_section)
