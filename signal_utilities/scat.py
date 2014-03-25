#!/usr/bin/python -tt
import numpy as np
from operator import add

#todo, derive this from a base class 'Feature' and define methods like flatten and reduce for that
class Scat(object):
    """
    Scattering class for storing and retreiving and performing operations on scattering 
    coefficients.
    """
    def __init__(self,root_node,int_orientations,int_levels,depth):
        """
        Class constructor for Scattering object,
        coeffs_index is a dictionary of path lists used to look-up
        and index in the coeff_tree
        """
        self.root_node = root_node
        self.int_orientations = int_orientations
        self.max_transform_levels = int_levels
        self.depth = depth
        self.input_dim = root_node.data.ndim #dimension of the input signal
        self.depth_first_nodes = None
        self.breadth_first_nodes = None
        
    def retrieve(self,subband_path):   
        """Given a list (subband_path), return a node corresponding
        to this path
        """
        if subband_path[0]!=0:
            ValueError('every subband path should begin with 0 ' + str(subband_path))
        node = self.root_node
        for level_index,path_index in enumerate(subband_path[1:]):
            if node.children!=[]:
                node = node.children[path_index]
            else:
                Warning('subband path exceeds depth of tree '  + 
                        str(subband_path) + '...stopping at ' + str(level_index))
        return node         

    def get_nodes(self,traversal_method='breadth'):
        """returns a list of the nodes in this tree, using the
        traversal method specified: '
        """
        if traversal_method=='breadth':
            if self.breadth_first_nodes == None:
                self.breadth_first_nodes = []
                parent_nodes = [self.root_node]
                parent_nodes_next=[]
                while parent_nodes!=[]:
                    for parent_node in parent_nodes:
                        self.breadth_first_nodes.append(parent_node)
                        parent_nodes_next += parent_node.children
                    parent_nodes = parent_nodes_next
                    parent_nodes_next = []    
            return self.breadth_first_nodes
        else:
            ValueError('unsupported traversal ' + traversal_method)
            
    def flatten(self,traversal_method='breadth'):
        """returns a d+1-D numpy array, with each of the nodes concatenated along
        a new axis, each node having a d-dimensional data member.
        """
        nodes_list = self.get_nodes(traversal_method)
        dims = nodes_list[0].get_data().ndim
        return np.concatenate(
            [node.get_data()[...,np.newaxis] for node in nodes_list],axis=self.input_dim)

    def reduce(self,traversal_method='breadth'):
        """Sums each of arrays returned by self.flatten to produce a single vector, 
        with each element corresponding to a path taken to a set of scattering
        coefficients
        """
        sum_dims = tuple([j for j in xrange(self.input_dim)])
        return np.sum(self.flatten(traversal_method),axis=sum_dims)