#!/usr/bin/python -tt
import sklearn.metrics as sm

from py_utils.results.metric import Metric

import pdb

class ClassificationMetric(Metric):
    """
    Base class for defining a metric
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
        else:
            raise ValueError('unsupported classification metric ' + self.metric_type)    

    def update(self,dict_in):
        pdb.set_trace()
        self.data.append(self.met_fun(dict_in['y_truth'],dict_in['y_pred']))
        super(ClassificationMetric,self).update(value)
        
    class Factory:
        def create(self,ps_params,str_section):
            return ClassificationMetric(ps_params,str_section)
