#!/usr/bin/python -tt
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt

from py_utils.results.metric import Metric

import pdb

class ClassificationMetric(Metric):
    """
    This class serves as an interface between scikitlearns metrics
    and the Metric class.
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for ClassificationMetric.
        """       
        super(ClassificationMetric,self).__init__(ps_params,str_section)
        self.metric_type = self.get_val('metrictype',False)
        if self.metric_type=='accuracy':
            self.met_fun = sm.accuracy_score
        elif self.metric_type=='f1':
            self.met_fun = sm.f1_score
        elif self.metric_type=='fbeta':
            self.met_fun = sm.fbeta_score
        elif self.metric_type=='hamming':
            self.met_fun = sm.hamming_loss
        elif self.metric_type=='jaccardsimilarity':
            self.met_fun = sm.jaccard_similarity_score
        elif self.metric_type=='precisionrecallfscoresupport':
            self.met_fun = sm.precision_recall_fscore_support
        elif self.metric_type=='precision':
            self.met_fun = sm.precision_score
        elif self.metric_type=='recall':
            self.met_fun = sm.recall_score
        elif self.metric_type=='zeroone':
            self.met_fun = sm.zero_one_loss
        elif self.metric_type=='confusionmatrix':
            self.met_fun = sm.confusion_matrix
        else:
            raise ValueError('unsupported classification metric ' + self.metric_type)    
        
    def plot(self):
        if self.metric_type!='confusionmatrix':
            super(ClassificationMetric,self).plot()
        else: 
            plt.figure(self.figure_number)   
            plt.matshow(self.data[-1])
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            num_classes=len(self.class_labels)
            if num_classes>0:
                plt.xticks(np.arange(len(self.class_labels)),self.class_labels)
                plt.yticks(np.arange(len(self.class_labels)),self.class_labels)
                for ix in xrange(num_classes):
                    for iy in xrange(num_classes):
                        plt.annotate(self.data[-1][ix][iy],(iy,ix))
            
    def update(self,dict_in):
        #replicate this metric for the number of testing instances
        self.data.append(self.met_fun(dict_in['y_truth'],dict_in['y_pred']))
        if dict_in.has_key('class_labels'):
            self.class_labels=dict_in['class_labels']
        else:
            self.class_labels=[]
        super(ClassificationMetric,self).update()
        
    class Factory:
        def create(self,ps_params,str_section):
            return ClassificationMetric(ps_params,str_section)
