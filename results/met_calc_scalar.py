#!/usr/bin/python -tt
from py_utils.results.metric import Metric
import numpy as np
from numpy.linalg import norm

import pdb

class CalcScalar(Metric):
    """
    CalcScalar metric class, for storing a single number or label vs iteration or sample index.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for ISNR.
        """       
        super(CalcScalar,self).__init__(ps_parameters,str_section)        
        
    def update(self,dict_in):
        """
        Expects a single value or array. If array, store the whole vector or list and stop.
        """
        #one time updates
        if self.key == 'nu_sq_convex':
            epsilon_sq = dict_in['epsilon_sq'][len(self.data)]
            rhsmin = dict_in['w_n'][0].energy().flatten()
            rhsmin = rhsmin + 4*epsilon_sq + 4*epsilon_sq**2/rhsmin 
            rhsmin = np.min(rhsmin)
            lhsmin = np.min(dict_in['alpha'])
            value = 1.0/8*lhsmin*rhsmin
        elif self.key == 'penalty_fun':
            if self.data == []:
                self.y = dict_in['y'].flatten()
                self.z = dict_in['w_n'][0] * 1
                self.get_legend_info()
            epsilon_sq = dict_in['epsilon_sq'][len(self.data)]
            nu_sq = dict_in['nu_sq'][len(self.data)]
            x_n_current = (~dict_in['W']) * dict_in['w_n'][0]
            dict_in['H'].set_output_fourier(False)
            x_n_current = (dict_in['H']) * x_n_current
            S_n_current = np.real(dict_in['S_n'].flatten())
            data_fidelity = norm(self.y - x_n_current.flatten(),2)**2
            sparsity = nu_sq * (np.dot(dict_in['w_n'][0].energy().flatten(),S_n_current) 
                                 - np.sum(np.log(S_n_current)) + epsilon_sq*np.sum(S_n_current)
                                 - 2.0 * 2.0 * S_n_current.size * np.log(epsilon_sq / 2))
            w_minus_z = dict_in['w_n'][0] + (self.z * (-1))
            #this should be 0 for the first update as there is no 'previous estimate' of w_n
            
            mm_term1 = np.sum((w_minus_z.energy() * dict_in['alpha']).flatten())
            mm_term2 = (~dict_in['W']) * w_minus_z
            dict_in['H'].set_output_fourier(False)
            mm_term2 = -(norm((dict_in['H'] * mm_term2).flatten(),2)**2)

            mm_term_tot = mm_term1 + mm_term2
            self.z = dict_in['w_n'][0] * 1 #copy this iterate for the next estimate
            totalJ = data_fidelity + sparsity + mm_term_tot
            value = np.asarray([data_fidelity, sparsity, mm_term_tot, totalJ])
            #msist_penalty_fun
        super(CalcScalar,self).update(value)    
    def save(self,strPath='/home/outputimage/'):
        self.save_csv(strPath)
            
    def get_legend_info(self):
        self.legend_labels = [r'$\|\mathbf{\Phi w}_n - \mathbf{y}\|_2^2$',
                              r'Sparsity',
                              r'MM']
        self.legend_cols = 1
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return CalcScalar(ps_parameters,str_section)
