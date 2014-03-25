#!/usr/bin/python -tt
import numpy as np
from operator import add
from numpy import concatenate as cat
from copy import deepcopy
import theano
from theano import tensor as T
from py_utils.signal_utilities.sig_utils import downsample_slices
import itertools as it

import pdb

class Node(object):
    """
    Node class for wrapping an object. Follows a parent + children model.
    """
    def __init__(self,wrapped_class):
        """
        Class constructor for Node
        """
        self.wrapped_class = wrapped_class
        self.children = []
        self.data = None

    def __getattr__(self,attr):
        if self.wrapped_class != None:
            orig_attr = self.wrapped_class.__getattribute__(attr)
            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    if result == self.wrapped_class:
                        return self
                    return result
                return hooked
            else:
                return orig_attr
        else:     
            return None

    def set_data(self,data):
        self.data = data

    def get_data(self):
        return self.data
    
    def set_children(self,children):
        self.children = children    

    def delete_wrapped_instance(self):
        """Setting the wrapped_class to None, and relying on garbage collection to 
        free resources. Explicitly deleting the wrapped class can lead to undefined
        behavior in the __getattr__ method otherwise.
        """
        
        self.wrapped_class = None