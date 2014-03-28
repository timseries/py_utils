#!/usr/bin/python -tt
import numpy as np
from operator import add

#todo, derive this from a base class 'Feature' and define methods like flatten and reduce for that
class Scat(object):
    """
    Scattering class for storing and retreiving and performing operations on scattering 
    coefficients.
    """
    def __init__(self,root_node=None,int_orientations=None,int_levels=None,depth=None):
        """
        Class constructor for Scattering object,
        coeffs_index is a dictionary of path lists used to look-up
        and index in the coeff_tree
        """
        self.root_node = root_node
        self.int_orientations = int_orientations
        self.max_transform_levels = int_levels
        self.depth = depth
        self.depth_first_nodes = None
        self.breadth_first_nodes = None
        
    def retrieve(self,subband_path):   
        """Given a list (subband_path), return a node corresponding
        to this path
        """
        if subband_path[0]!=0:
            raise ValueError('every subband path should begin with 0 ' + str(subband_path))
        node = self.root_node
        for level_index,path_index in enumerate(subband_path[1:]):
            if node.children!=[]:
                node = node.children[path_index]
            else:
                raise Warning('subband path exceeds depth of tree '  + 
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
            raise ValueError('unsupported traversal ' + traversal_method)
            
    def flatten(self,traversal_method='breadth'):
        """returns a d+1-D numpy array, with each of the nodes concatenated along
        a new axis, each node having a d-dimensional data member.
        """
        nodes_list = self.get_nodes(traversal_method)
        dims = nodes_list[0].get_data().ndim
        return np.concatenate(
            [node.get_data()[...,np.newaxis] for node in nodes_list],axis=dims)
    
    def gen_image(self,traversal_method='breadth'):
        """generates an image (numpy array) by traversing the scat tree
        """
        #flatten out the tree
        nodes_list = self.get_nodes(traversal_method)
        #get all of the subband path lists, put in a list
        subband_paths=[node.path for node in nodes_list]
        #build a list of subband path lists, each element corresponding to a scattering depth
        min_path_len=min([len(subband_paths) for subband_path in subband_pathss])
        max_path_len=max([len(subband_paths) for subband_path in subband_pathss])
        #filter by subband path length
        subband_paths_by_level=[sorted([subband_path for subband_path in subband_paths if len(subband_path) == pathlen])
                                for pathlen in xrange(min_path_len,max_path_len+1)]
        #initialize montage
        thumbnail_width = nodes_list[0].data.shape[0]
        thumbnail_height = nodes_list[0].data.shape[1]
        montage_cols = 0
        montage_width = 0
        #now loop and build the montage
        return np.zeros(10) #todo('FIX')

    @staticmethod
    def reduce(flattened_scat,method='sum'):
        """Sums each of arrays returned by self.flatten to produce a single vector, 
        with each element corresponding to a path taken to a set of scattering
        coefficients
        """
        sum_dims = tuple([j for j in xrange(flattened_scat.ndim-1)])
        if method=='sum' or method=='':
            return np.sum(flattened_scat,axis=sum_dims,dtype='float32')
        if method=='average':
            return np.average(flattened_scat,axis=sum_dims,dtype='float32')
        else:
            raise ValueError('no such reduce method in Scat')
