#!/usr/bin/python -tt
import scipy.ndimage
import numpy as np
from numpy.ma.core import exp, sqrt
from numpy import arange, asarray, ndarray
from scipy.constants.constants import pi
import ImageOps
from py_utils.results.metric import Metric
"""
This class module computes the Structured Similarity Image Metric (SSIM)


..codeauthor:: Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr, http://isit.u-clermont1.fr/~anvacava Created on 21 nov. 2011
..codeauthor:: Modified by Christopher Godfrey, on 17 July 2012 (lines 32-34)
..codeauthor:: Modified by Jeff Terrace, starting 29 August 2012
..codeauthor:: Timothy Roberts <timothy.daniel.roberts@gmail.com>, extended to 3D
"""
class SSIM(Metric):
    """
    Base class for defining a metric
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(SSIM,self).__init__(ps_parameters,str_section)
        self.x = None #ground truth
        self.K = self.get_val('k',True)
        if np.all(self.K == np.zeros(2)):
            self.K = np.array([.01, .03])
        self.gaussian_kernel_sigma = self.get_val('gaussiankernelsigma',True) 
        if self.gaussian_kernel_sigma == 0:
            self.gaussian_kernel_sigma = 1.5
        self.gaussian_kernel_width = self.get_val('gaussiankernelwidth',True)
        if self.gaussian_kernel_width == 0:
            self.gaussian_kernel_width = 11
            
    def update(self,dict_in):
        if self.data == []:
            self.x = dict_in['x']
        x_n = dict_in['x_n']
        if x_n.ndim == 2:
            value = self.compute_ssim(self.x, x_n)
        elif x_n.ndim == 3:    
            value = self.compute_ssim_3d(self.x, x_n)
        else:
            raise Exception('unsupported number of dimensions in x_n')
        self.data.append(value)
        super(SSIM,self).update()

    def _to_grayscale(im):
        """
        Convert PIL image to numpy grayscale array and numpy alpha array
        @param im: PIL Image object
        @return (gray, alpha), both numpy arrays
        """
        gray = asarray(ImageOps.grayscale(im)).astype(np.float)
        imbands = im.getbands()
        if 'A' in imbands:
            alpha = asarray(im.split()[-1]).astype(np.float)
        else:
            alpha = None
        return gray, alpha

    def convolve_gaussian_2d(self, image, gaussian_kernel_1d):
        result = scipy.ndimage.filters.correlate1d(image, gaussian_kernel_1d, axis = 0)
        result = scipy.ndimage.filters.correlate1d(result, gaussian_kernel_1d, axis = 1)
        return result

    def compute_ssim_3d(self,im1, im2):
        if im1.shape != im2.shape:
            raise Exception('comparison volumes unequal dimensions')
        else:
            ssim_mean = 0
            ssim_min = np.inf
            for i in arange(im1.shape[2]):
                ssim = self.compute_ssim(im1,im2)
                ssim_mean += ssim
                ssim_min = np.min([ssim_min,ssim])
            ssim_mean = ssim_mean / im1.shape[2]
        return ssim_mean, ssim_min
                
    def compute_ssim(self,im1, im2):
        """
        The function to compute SSIM
        @param im1: PIL Image object, or grayscale ndarray
        @param im2: PIL Image object, or grayscale ndarray
        @return: SSIM float value
        """
    
        # 1D Gaussian kernel definition
        gaussian_kernel_1d = np.ndarray((self.gaussian_kernel_width))
        mu = int(self.gaussian_kernel_width / 2)
        
        #Fill Gaussian kernel
        for i in xrange(self.gaussian_kernel_width):
            gaussian_kernel_1d[i] = (1 / (sqrt(2 * pi) * (self.gaussian_kernel_sigma))) * \
              exp(-(((i - mu) ** 2)) / (2 * (self.gaussian_kernel_sigma ** 2)))

        # convert the images to grayscale
        if im1.__class__.__name__ == 'Image':
            img_mat_1, img_alpha_1 = _to_grayscale(im1)
            # don't count pixels where both images are both fully transparent
            if img_alpha_1 is not None:
                img_mat_1[img_alpha_1 == 255] = 0
        else:
            img_mat_1 = im1
        if im2.__class__.__name__ == 'Image':    
            img_mat_2, img_alpha_2 = _to_grayscale(im2)
            if img_alpha_2 is not None:
                img_mat_2[img_alpha_2 == 255] = 0
        else:
            img_mat_2 = im2
      
        #Squares of input matrices
        img_mat_1_sq = img_mat_1 ** 2
        img_mat_2_sq = img_mat_2 ** 2
        img_mat_12 = img_mat_1 * img_mat_2
            
        #Means obtained by Gaussian filtering of inputs
        img_mat_mu_1 = self.convolve_gaussian_2d(img_mat_1, gaussian_kernel_1d)
        img_mat_mu_2 = self.convolve_gaussian_2d(img_mat_2, gaussian_kernel_1d)
      
        #Squares of means
        img_mat_mu_1_sq = img_mat_mu_1 ** 2
        img_mat_mu_2_sq = img_mat_mu_2 ** 2
        img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2
      
        #Variances obtained by Gaussian filtering of inputs' squares
        img_mat_sigma_1_sq = self.convolve_gaussian_2d(img_mat_1_sq, gaussian_kernel_1d)
        img_mat_sigma_2_sq = self.convolve_gaussian_2d(img_mat_2_sq, gaussian_kernel_1d)
      
        #Covariance
        img_mat_sigma_12 = self.convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)
      
        #Centered squares of variances
        img_mat_sigma_1_sq -= img_mat_mu_1_sq
        img_mat_sigma_2_sq -= img_mat_mu_2_sq
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12
        
        #set k1,k2 & c1,c2 to depend on L (width of color map)
        #l = 255
        L = np.max(img_mat_2.flatten())-np.min(img_mat_2.flatten())
        k_1 = self.K[0]
        c_1 = (k_1 * L) ** 2
        k_2 = self.K[1]
        c_2 = (k_2 * L) ** 2
      
        #Numerator of SSIM
        num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
      
        #Denominator of SSIM
        den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
          (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
      
        #SSIM
        ssim_map = num_ssim / den_ssim
        index = np.average(ssim_map)

        return index

    class Factory:
        def create(self,ps_parameters,str_section):
            return SSIM(ps_parameters,str_section)
