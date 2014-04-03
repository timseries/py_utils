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
        """generates dict for use in an output_montage object
        """
        #flatten out the tree
        nodes_list = self.get_nodes(traversal_method)
        #get all of the subband path lists, put in a list
        subband_paths=[node.path for node in nodes_list]
        #build a list of subband path lists, each element corresponding to a scattering depth
        min_path_len=min([len(subband_paths) for subband_path in subband_paths])
        max_path_len=max([len(subband_paths) for subband_path in subband_paths])
        #filter by subband path length
        subband_paths_by_level=[sorted([subband_path for subband_path in subband_paths if len(subband_path) == pathlen])
                                for pathlen in xrange(min_path_len,max_path_len+1)]
        #initialize the montage dictionary
        montage_dict={}
        montage_dict['thumbnail_height']=nodes_list[0].get_data().shape(0)
        montage_dict['thumbnail_width']=nodes_list[0].get_data().shape(1)
        montage_dict['thumbnail_columns']=self.int_orientations**(self.depth-1)
        montage_dict['thumbnail_rows']=self.depth*self.max_transform_levels
        montage_dict['ls_images']=[]
        montage_dict['ls_strings']=[]
        montage_dict['ls_locs']=[]
        #now loop and build the montage lists of images and locations
        for level_ix,subband_path_by_level in enumerate(subband_paths_by_level):
            level_row_offset=level_ix*montage_dict['thumbnail_height']*self.max_transform_levels
            prev_subband_path=subband_path_by_level[0]
            block_col_index=0
            for block_ix,subband_path in enumerate(subband_path_by_level):
                node = self.retrieve(subband_path)
                row_offset=((node.get_scale()-1)*montage_dict['thumbnail_height']
                                  +level_row_offset)
                if prev_subband_path[-2]!=subband_path[-2]:#parent changed
                    block_col_index+=1
                if level_ix>1:                     
                    level_col_offset=(block_col_index)*self.int_orientations*montage_dict['thumbnail_width']
                    block_col_offset=mod((subband_path[-1]-1),self.int_orientations)*montage_dict['thumbnail_width']
                else:    
                    level_col_offset=0
                    block_col_offset=0
                prev_subband_path=subband_path    
                col_offset=level_col_offset+col_offset
                montage_dict['ls_images'].append(node.get_data())
                montage_dict['ls_strings'].append(str(subband_path))
                montage_dict['ls_locs'].append(np.array([row_offset,col_offset]))
        return montage_dict

    @staticmethod
    def reduce(flattened_scat,method='sum'):
        """Sums each of arrays returned by self.flatten to produce a single vector, 
        with each element corresponding to a path taken to a set of scattering
        coefficients
        """
        sum_dims = tuple([j for j in xrange(flattened_scat.ndim-1)])
        if method=='sum' or method=='':
            return np.sum(flattened_scat,axis=sum_dims,dtype='float64')
        elif method=='average':
            # return np.asfarray(np.average(flattened_scat,axis=sum_dims)) #doesnt' work on old numpy
            normalizer=np.prod([flattened_scat.shape[j] for j in xrange(flattened_scat.ndim-1)])
            return np.sum(np.sum(flattened_scat,axis=0,dtype='float64'),axis=0,dtype='float64')/normalizer
        else:
            raise ValueError('no such reduce method ' + method + ' in Scat')
