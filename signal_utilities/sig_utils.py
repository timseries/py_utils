#!/usr/bin/python -tt
from operator import add
import numpy as np
from numpy import max as nmax, absolute, conj, arange, zeros, array, median, pad
from numpy.fft import fftn, ifftn
from numpy.linalg import norm, inv
from numpy.random import normal, rand, seed, poisson
from scipy.stats.mstats import mode
from scipy.sparse import csr_matrix

import itertools
from collections import defaultdict

import pdb

def nd_impulse(ary_size):
    ary_impulse = zeros(ary_size)
    ary_impulse[tuple(array(ary_impulse.shape)/2)] = 1
    return ary_impulse

def spectral_radius(op_transform, op_modality, tup_size, method='spectrum'):
    """Compute the specral radius (the dominant eigenvalues) of the composition of op_transform and op_modality.
    :param op_transform: a transform operator which returns a ws object
    :param op_modality: some linear operator, 
     assumes this is diagonalizable by the fft (e.g. convolution)
    :param tup_size: size of impulse used...
    :param method: the available methods for computing these weights (spectrum or power_iteration)
    :returns ary_lambda_alpha, a vector of subband weights which upperbounds the spectral radius
    
    .. codeauthor:: Timothy Roberts <timothy.daniel.roberts@gmail.com>, 2013
    """
    if method=='spectrum':
        ary_impulse = nd_impulse(tup_size)
        ws_w = op_transform * ary_impulse
        ary_alpha = zeros(ws_w.int_subbands)
        ary_temp = op_modality * ary_impulse # unused result, just to initialize with correct size
        ary_ms = op_modality.get_spectrum()
        if tup_size != ary_ms.shape:
            ary_impulse = nd_impulse(ary_ms.shape)
            ws_w = op_transform * ary_impulse
        ary_ms = ary_ms.flatten()
        ary_subbands = arange(ws_w.int_subbands)
        ary_ss = zeros([ary_ms.shape[0],ws_w.int_subbands],dtype='complex64')
        ary_alpha = zeros(ws_w.int_subbands,)
        #store the inband psd's (and conjugate)
        for s in ary_subbands:
            ary_ss[:,s] = fftn(~op_transform * ws_w.one_subband(s)).flatten()
        #compute the inband and crossband psd maxima
        for s in ary_subbands:
            for c in ary_subbands:
                 ary_alpha[s] += nmax(absolute(conj(ary_ms * ary_ss[:,s]) * ary_ms * ary_ss[:,c]))
        #return the alpha array, upper-bounded to 1                      
        ary_alpha = np.minimum(ary_alpha,1.0)
        
    elif method=='power_iteration':
        W = op_transform
        Phi = op_modality
        ws_rv = op_transform * rand(*tup_size) #random vector
        ary_eigs = np.zeros(ws_rv.int_subbands)
        for s in xrange(ws_rv.int_subbands):
            ws_ss = ws_rv.one_subband(s)
            i = 0
            while i < 30:
                ws_ss = A(W,Phi,ws_ss) #to image domain
                ws_ss = At(W,Phi,ws_ss) #back to wavelet domain
                ws_ss_norm = norm(ws_ss.flatten(),2)
                for s2 in xrange(ws_ss.int_subbands):
                    ws_ss.set_subband(s2,ws_ss.get_subband(s2)/ws_ss_norm)
                i += 1    
                ws_ss = ws_ss.one_subband(s)    
            AtAws_ss = A(W,Phi,ws_ss)
            AtAws_ss = At(W,Phi,AtAws_ss).flatten()
            print np.sum(ws_ss.flatten().transpose() * AtAws_ss)
            print ws_ss_norm
            ary_eigs[s] = np.sum(ws_ss.flatten().transpose() * AtAws_ss) / ws_ss_norm
            print ary_eigs[s]
            print ary_eigs
        ary_alpha = np.minimum(ary_eigs,1.0)
    else:
        raise ValueError('method ' + method + ' unsupported')
                
    print ary_alpha
    return ary_alpha


def A(op_transform,op_modality,mcand):
    return op_modality * (~op_transform * mcand)

def At(op_transform,op_modality,mcand):
    return op_transform * (~op_modality * mcand)


def circshift(ary_input, tup_shifts):
    """Shift multi-dimensional array circularly.
  
    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.
  
    Parameters
    
    ary_input : ndarray to shift.
  
    s : tuple of int
       A tuple of integer scalars where the N-th element specifies the
       shift amount for the N-th dimension of input array. If an element
       is positive, the values of `a` are shifted down (or to the
       right). If it is negative, the values of `a` are shifted up (or
       to the left).

    Returns
    -------
    y : ndarray
       The shifted array (elements are copied)
  
    Examples
    --------
    >>> circshift(np.arange(10), 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
  
    """
    # Initialize array of indices
    idx = []
  
    # Loop through each dimension of the input matrix to calculate
    # shifted indices
    for dim in range(ary_input.ndim):
        length = ary_input.shape[dim]
        try:
            shift = tup_shifts[dim]
        except IndexError:
            shift = 0  # no shift if not specify
  
        index = np.mod(np.array(range(length),
                                ndmin=ary_input.ndim) - shift,
                                length)

        shape = np.ones(ary_input.ndim,dtype='uint16')
        shape[dim] = ary_input.shape[dim]
        index = np.reshape(index, shape)
        idx.append(index.astype(int))
  
    # Perform the actual conversion by indexing into the input matrix
    return ary_input[idx]

def noise_gen(noise_params):
    '''
    A funtion to compute realizations of an additive noise process
    Expects (in noise_params dictionary):
    seed, variance, distribution, mean, size, (interval, for noise with finite support pdf)
    '''
    int_seed = noise_params['seed']
    dbl_mean = noise_params['mean']
    dbl_variance = noise_params['variance']
    tup_size = noise_params['size']

    #set the seed, always set the seed!
    seed(int_seed)
    if noise_params['distribution'] == 'gaussian':
        ary_noise = normal(dbl_mean, np.sqrt(dbl_variance), tup_size)
    elif noise_params['distribution'] == 'uniform':
        ary_interval = noise_params['interval']
        ary_noise = (ary_interval[1] - ary_interval[0]) * rand(tup_size) + ary_interval[0]
    elif noise_params['distribution'] == 'poisson':     
        ary_noise = poisson(lam=noise_params['ary_mean'])
    else:
        raise Exception('unsupported noise distribution: ' + noise_params['distribution'])
    return ary_noise

def colonvec(ary_small, ary_large):
    """
    Compute the indices used to pad/crop the results of applying the fft with augmented dimensions
    ary_small and ary_large are shape tuples which define the bounds of the cropping region
    """
    int_max = np.maximum(len(ary_small),len(ary_large))
    indices = [np.s_[int(ary_small[i]-1):int(ary_large[i])] for i in np.arange(int_max)]
    return indices

def pad_center(ary_input, tup_new_size):
    """
    Returns a new array of size tup_new_size with ary_input in the center
    """
    if ((ary_input.shape > tup_new_size) or 
        (ary_input.ndim != len(tup_new_size))): #one of the dimensions is too big
        raise ValueError('cannot pad ary_input using new shape ' + 
                         str(tup_new_size) + ', input shape is ' + 
                         ary_input.shape)
    else: #inputs are OK
        pad_sz=(np.array(tup_new_size)-np.array(ary_input.shape))/2
        ls_pad_size_axis=[(pad_sz[j],pad_sz[j]) for j in xrange(pad_sz.size)]
        return np.pad(ary_input,ls_pad_size_axis,mode='constant',constant_values=0)

def get_neighborhoods(ary_input,n_size):
    """
    Given an array, return the neighborhoods of size n_size along a new axis.

    The return value is a tuple of values and a logical mask. The values are the elements of the neighborhood,
    and the mask tells you which (since not all neighborhoods are the same size) elements are valid

    Here, x's show a neighborhood of coefficient 'o' size 1 for a 2-d array:
    a b c 
    d o e
    f g h

    The function would return 9 arrays of the same size of 1. values and 2. masks. Here is the first of 1 arrays
    value:
    o e d 
    g h f
    b c a
    mask:
    True True False
    True True False
    False False False

    This makes it convenient for performing vector operations (e.g. average) on the neighborhoods, which is the whole reason for writing this function.
    
    """
    #calculate shifts
    dims = ary_input.ndim
    iter_dims = xrange(dims)
    grid = np.mgrid[[slice(-n_size,n_size+1 ,None) for i in iter_dims]]
    shifts = np.vstack([grid[i,...].flatten() for i in iter_dims]).transpose()
    iter_shifts = xrange(len(shifts))
    neighborhoods = [circshift(ary_input,tuple(shifts[i])) for i in iter_shifts]
    #now we have to zeroize the rows,columns,etc where edges have been shifted in
    for i in iter_shifts:
        ls_slices = shift_slices(shifts[i])
        if ls_slices!=[]:
            for j in xrange(len(ls_slices)):             
                neighborhoods[i][ls_slices[j]]=np.nan
    neighborhoods = np.concatenate([neighborhoods[i][...,np.newaxis] 
                                    for i in iter_shifts],axis=dims)
    mask = ~np.isnan(neighborhoods)
    neighborhoods[~mask]=0
    return tuple([neighborhoods,mask])

def shift_slices(shift):
    """calculate the slices corresponding to a given set of shifts
    e.g. shift=(-1,-1) would give a list of lists of slices : [[slice(-1,None,None),slice(None,None,None)],
                                                              [slice(None,None,None),slice(-1,None,None)]]
    """
    ls_slices = []
    iter_shifts = xrange(len(shift))
    if np.all(shift==0): #just return empty list for indexing nothing
        return []
    eslice = slice(None,None,None)
    for k in iter_shifts:
        if shift[k]!=0:
            slices = []
            for j in iter_shifts:
                if j==k:
                    if shift[k]<0:
                        sl=slice(shift[k],None,None)
                    else: #shift[k]>0:    
                        sl=slice(None,shift[k],None)
                else:
                    sl=eslice        
                slices.append(sl)
            ls_slices.append(slices)
    return ls_slices            

def downsample_slices(int_dim):
    """Return a list of list of slices used for downsampling by a factor of 2
    in every dimension, for each of 2**int_dim offsets
    """
    num_slices = 2**int_dim
    return [[slice(dec_to_bin(j,int_dim)[i],None,2) for i in xrange(int_dim)] for j in xrange(num_slices)]
            
def dec_to_bin(int_decimal,width=None):
    """returns int_decimal as a list of binary digits, with the lsbit
    being the first element, with msb padding
    """ 
    bin_string = bin(int_decimal)
    bin_width = len(bin_string)-2
    if width == None:
        width=bin_width
    return [int(bin_string[i]) for i in xrange(bin_width+1,1,-1)] + (width-bin_width)*[0] 
    
def crop_center(ary_signal, tup_crop_size):
    '''Crop ary_signal symmetrically to be tup_crop_size
    '''
    ary_half_difference = (array(ary_signal.shape) - array(tup_crop_size)) / 2
    return ary_signal[colonvec(ary_half_difference+1, ary_half_difference+array(tup_crop_size))]

def gaussian(shape=(3,3),sigma=(0.5,0.5)):
    """
    2D gaussian - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma[0]*sigma[1]) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def upsample(ary_input,factor=2,method='shiftadd'):
    ary_upsampled = np.zeros(np.array(ary_input.shape)*factor)
    if method=='shiftadd':
        if ary_upsampled.ndim!=2:
            raise ValueError('dimensions other than 2 not supported')
        y1=np.zeros((ary_upsampled.shape[0],ary_input.shape[1]))
        y1[slice(0,-1,2),...] = (0.75*ary_input 
                                    + 0.25*np.concatenate(
                                        [ary_input[slice(0,1),...], 
                                         ary_input[slice(0,-1),...]],axis=0))
        y1[slice(1,None,2),...] = (0.75*ary_input 
                                      + 0.25*np.concatenate(
                                            [ary_input[slice(1,None),...],
                                             ary_input[slice(-1,None),...]],axis=0))
        ary_upsampled[...,slice(0,-1,2)] = (0.75*y1 
                                               + 0.25*np.concatenate(
                                                   [y1[...,slice(0,1)],
                                                    y1[...,slice(0,-1)]],axis=1))
        ary_upsampled[...,slice(1,None,2)] = (0.75*y1
                                               + 0.25*np.concatenate(
                                                   [y1[...,slice(1,None)],
                                                   y1[...,slice(-1,None)]],axis=1))
        return ary_upsampled
    else:
        raise ValueError('unsupported upsample method ' + method)
    
def flatten_list(ls_ary):
    '''Flatten a list of objects which have the flatten() method
    Assumes each element is the same size
    '''
    temp_ary=ls_ary[0].flatten()
    ary_size=temp_ary.size
    vec_ix=0
    output_ary=np.zeros(ary_size*len(ls_ary),dtype=temp_ary.dtype)
    for ary_unflat in ls_ary:
        output_ary[vec_ix:vec_ix+ary_size]=ary_unflat.flatten()
        vec_ix+=ary_size
    return output_ary

def flatten_list_to_csr(ls_ary):
    '''Flatten a list of objects which have the flatten() method to a 
    csr object. Assumes each list element is the same size
    '''
    ary_flattened=flatten_list(ls_ary)
    ary_flattened_sz=ary_flattened.size
    ary_diag=np.nonzero(ary_flattened)[0]
    return csr_matrix((ary_flattened[ary_flattened!=0],
                       (ary_diag,ary_diag)), 
                       shape=(ary_flattened_sz,ary_flattened_sz))

def unflatten_list(ary_input,num_partitions):
    '''Split ary_input into num_partitions equal-sized arrays and store 
    these in a list. Undoes flatten_list if the input to flatten_list is a list of
    equal-sized arrays.
    '''
    int_part_size=ary_input.size/num_partitions
    ls_ary=[]
    vec_ix=0
    for partition in xrange(num_partitions):
        ls_ary.append(ary_input[vec_ix:vec_ix+int_part_size])
        vec_ix+=int_part_size
    return ls_ary

def inv_block_diag(csr_bdiag, dict_in=None):
    '''input is a square sparse (csr) block diagonal matrix (non-overlapping)
    which may or may not be permuted. This linear-time function forms each of the blocks
    separately, inverts them, and then stores the entries back into a new csr_bdiag
    ouput. Fancy indexing in csr matrices is expensive and unsupported as of scipy .14
    '''
    #check to see if this block_diagonal matrix structure has been memoized
    if dict_in!=None and dict_in.has_key('dict_bdiag'):
        dict_bdiag=dict_in['dict_bdiag']
        csr_rows=dict_bdiag['csr_rows']
        csr_cols=dict_bdiag['csr_cols']
    else:
        dict_bdiag={}
        dict_in['dict_bdiag']=dict_bdiag
        csr_rows=np.nonzero(csr_bdiag)
        csr_cols=csr_rows[1]#this is not a mistake, just saving memory
        csr_rows=csr_rows[0]
        if dict_in.has_key('col_offset'):
            #it is assumed there that the column offset between entries in a given row
            #is always some multiple called col_offset
            col_offset=dict_in['col_offset']
            csr_clusters=csr_rows.copy()
            while np.max(csr_clusters)>=col_offset:
                csr_clusters[csr_clusters>=col_offset]-=col_offset
        else:
            #no clustering information (the column offset or otherewise) 
            #has been provided, so default to sorting by row order
            csr_clusters=np.zeros(csr_rows.size,dtype='uint8')
        #sort by the 'cluster' number
        #create a new variable sorted_temp for clustering together the coordinates 
        sorted_temp=sorted(zip(csr_clusters,csr_rows,csr_cols))
        csr_clusters=np.array([sorted_temp_pt[0] for sorted_temp_pt in sorted_temp],dtype='uint32')
        csr_rows=np.array([sorted_temp_pt[1] for sorted_temp_pt in sorted_temp],dtype='uint32')
        csr_cols=np.array([sorted_temp_pt[2] for sorted_temp_pt in sorted_temp],dtype='uint32')
        del sorted_temp
        #store the structure of this matrix
        dict_bdiag['int_max_block_sz']=mode(csr_rows)[1]
        dict_bdiag['csr_rows']=csr_rows
        dict_bdiag['csr_cols']=csr_cols
        #create a block size mask, speed things up using a hashtable
        dict_temp=defaultdict(lambda:0)
        for row_num in csr_rows:
            dict_temp[row_num]+=1
        block_sz_mask=np.array([dict_temp[row_num] for row_num in csr_rows],dtype='uint32')
        dict_bdiag['block_sizes']=sorted(np.unique(dict_temp.values()))
        del dict_temp
        #store index arrays for each block size
        for block_sz in dict_bdiag['block_sizes']:
            int_std=block_sz**2
            #blk_ix are indices within the block size mask which show the locations of elements corresponding to 
            #a given block size
            blk_ix=np.array(np.nonzero(block_sz_mask==block_sz)[0],dtype='uint32')#all of the block indices for csr_rows/cols
            #need to permute these block indices so that
            #the diagonal components to be in the correct locations
            #and we assume arrangement of off-diagonal elements doesn't matter
            #because they are all the same. without any additional info about csr_bdiag
            #this is the best we can do
            new_blk_ix=zip(csr_clusters[blk_ix],csr_rows[blk_ix],csr_cols[blk_ix],blk_ix)
            new_blk_ix=sorted(new_blk_ix,key=lambda x:(x[0],x[1],not(x[1]==x[2]),x[2]))
            new_blk_ix=np.array([new_blk_ix_pt[3] for new_blk_ix_pt in new_blk_ix],dtype='uint32')
            #now new_blk_ix has the diagonal elements in every block_sz position within each block
            #this is not correct, so we'll correct this in the following loop
            #at least now we have an efficient way to index the diagonal components in each block
            #the following offsets are with respect to the new_blk_indexing
            all_blk_offsets=set(np.arange(int_std))
            diag_new_blk_offsets=set(np.arange(0,int_std,block_sz))
            diag_blk_offsets=set(np.arange(0,int_std,block_sz+1))
            off_diag_new_blk_offsets=sorted(all_blk_offsets.difference(diag_new_blk_offsets))
            off_diag_blk_offsets=sorted(all_blk_offsets.difference(diag_blk_offsets))
            #assign the each of diagonal components to the correct locations in blk_ix
            for blk_offset,new_blk_offset in zip(diag_blk_offsets,diag_new_blk_offsets):
                blk_ix[blk_offset::int_std]=new_blk_ix[new_blk_offset::int_std]
            #assign the remaining off-diagonal components to the correct locations in blk_ix
            for blk_offset,new_blk_offset in zip(off_diag_blk_offsets,off_diag_new_blk_offsets):
                blk_ix[blk_offset::int_std]=new_blk_ix[new_blk_offset::int_std]
            dict_bdiag[str(block_sz)+'rows']=csr_rows[blk_ix]
            dict_bdiag[str(block_sz)+'cols']=csr_cols[blk_ix]
            dict_bdiag[block_sz]=blk_ix
    #done computing the indices used to extract the blocks from the csr_bdiag
    #now we can do the actual inverting, storing in a new vector new_csr_data
    new_csr_data=np.zeros(csr_rows.size,)
    for block_sz in dict_bdiag['block_sizes']:
        int_std=block_sz**2 #stride to extract the blocks, when necessary
        blk_csr_rows=dict_bdiag[str(block_sz)+'rows']
        blk_csr_cols=dict_bdiag[str(block_sz)+'cols']
        #when you fancy index into a csr mtx, you get a matrix, hence the funny indexing to convert to array
        ary_blk_v=np.asarray(csr_bdiag[blk_csr_rows,blk_csr_cols])[0,:]
        ary_blk_vsz=ary_blk_v.size
        if block_sz==1:
            ary_blk_v=1.0/ary_blk_v
        elif block_sz==2:    
            #for [[a b],[c d]]
            #can use accelerated 2x2 block matrix inversion 1/det(A)*[[a22 -a12],[-a21 a11]]
            #detA = 1/(a11a22-a12a21)
            #these are stored in the vector format as [a11 a12 a21 a22] since we've sorted by rows
            minor_mtx=np.zeros(ary_blk_vsz,)
            a11=ary_blk_v[0::int_std]
            a12=ary_blk_v[1::int_std]
            a21=ary_blk_v[2::int_std]
            a22=ary_blk_v[3::int_std]
            minor_mtx[0::int_std]=a22
            minor_mtx[1::int_std]=-a12
            minor_mtx[2::int_std]=-a21
            minor_mtx[3::int_std]=a11
            ary_blk_v=minor_mtx / np.repeat(a11*a22-a12*a21,int_std)
        elif block_sz==4:
            #these equations generated by using symbolic matrix inversion in matlab
            minor_mtx=np.zeros(ary_blk_vsz,)
            a11=ary_blk_v[0::int_std]
            a12=ary_blk_v[1::int_std]
            a13=ary_blk_v[2::int_std]
            a14=ary_blk_v[3::int_std]
            a21=ary_blk_v[4::int_std]
            a22=ary_blk_v[5::int_std]
            a23=ary_blk_v[6::int_std]
            a24=ary_blk_v[7::int_std]
            a31=ary_blk_v[8::int_std]
            a32=ary_blk_v[9::int_std]
            a33=ary_blk_v[10::int_std]
            a34=ary_blk_v[11::int_std]
            a41=ary_blk_v[12::int_std]
            a42=ary_blk_v[13::int_std]
            a43=ary_blk_v[14::int_std]
            a44=ary_blk_v[15::int_std]
            b11=a22*a33*a44+a23*a34*a42+a24*a32*a43-a22*a34*a43-a23*a32*a44-a24*a33*a42
            b12=a12*a34*a43+a13*a32*a44+a14*a33*a42-a12*a33*a44-a13*a34*a42-a14*a32*a43
            b13=a12*a23*a44+a13*a24*a42+a14*a22*a43-a12*a24*a43-a13*a22*a44-a14*a23*a42
            b14=a12*a24*a33+a13*a22*a34+a14*a23*a32-a12*a23*a34-a13*a24*a32-a14*a22*a33
            b21=a21*a34*a43+a23*a31*a44+a24*a33*a41-a21*a33*a44-a23*a34*a41-a24*a31*a43
            b22=a11*a33*a44+a13*a34*a41+a14*a31*a43-a11*a34*a43-a13*a31*a44-a14*a33*a41
            b23=a11*a24*a43+a13*a21*a44+a13*a23*a41-a11*a23*a44-a13*a24*a41-a14*a21*a43
            b24=a11*a23*a34+a13*a24*a31+a14*a21*a33-a11*a24*a33-a13*a21*a34-a14*a23*a31
            b31=a21*a32*a44+a22*a34*a41+a24*a31*a42-a21*a34*a42-a22*a31*a44-a24*a32*a41
            b32=a11*a34*a42+a12*a31*a44+a14*a32*a41-a11*a32*a44-a12*a34*a41-a14*a31*a42
            b33=a11*a22*a44+a12*a24*a41+a14*a21*a42-a11*a24*a42-a12*a21*a44-a14*a22*a41
            b34=a11*a24*a32+a12*a21*a34+a13*a22*a31-a11*a22*a34-a12*a24*a31-a14*a21*a32
            b41=a21*a33*a42+a22*a31*a43+a23*a32*a41-a21*a32*a43-a22*a33*a41-a23*a31*a42
            b42=a11*a32*a43+a12*a33*a41+a13*a31*a42-a11*a33*a42-a12*a31*a43-a13*a32*a41
            b43=a11*a23*a42+a12*a21*a43+a13*a22*a41-a11*a22*a43-a12*a23*a41-a13*a21*a42
            b44=a11*a22*a33+a12*a23*a31+a13*a21*a32-a11*a23*a32-a12*a21*a33-a13*a22*a31
            minor_mtx[0::int_std]=b11
            minor_mtx[1::int_std]=b12
            minor_mtx[2::int_std]=b13
            minor_mtx[3::int_std]=b14
            minor_mtx[4::int_std]=b21
            minor_mtx[5::int_std]=b22
            minor_mtx[6::int_std]=b23
            minor_mtx[7::int_std]=b24
            minor_mtx[8::int_std]=b31
            minor_mtx[9::int_std]=b32
            minor_mtx[10::int_std]=b33
            minor_mtx[11::int_std]=b34
            minor_mtx[12::int_std]=b41
            minor_mtx[13::int_std]=b42
            minor_mtx[14::int_std]=b43
            minor_mtx[15::int_std]=b44
            ary_blk_v=minor_mtx / np.repeat(  a11*a22*a33*a44+a11*a23*a34*a42+a11*a24*a32*a43
                                             +a12*a21*a34*a43+a12*a23*a31*a44+a12*a24*a33*a41
                                             +a13*a21*a32*a44+a13*a22*a34*a41+a13*a24*a31*a42
                                             +a14*a21*a33*a42+a14*a22*a31*a43+a14*a23*a32*a41
                                             -a11*a22*a34*a43-a11*a23*a32*a44-a11*a24*a33*a42
                                             -a12*a21*a33*a44-a12*a23*a34*a41-a12*a24*a31*a43
                                             -a13*a21*a34*a42-a13*a22*a31*a44-a13*a24*a32*a41
                                             -a14*a21*a32*a43-a14*a22*a33*a41-a14*a23*a31*a42,int_std)
            del minor_mtx
        elif block_sz==5:
            minor_mtx=np.zeros(ary_blk_vsz,)
            a11=ary_blk_v[0::int_std]
            a12=ary_blk_v[1::int_std]
            a13=ary_blk_v[2::int_std]
            a14=ary_blk_v[3::int_std]
            a15=ary_blk_v[4::int_std]
            a21=ary_blk_v[5::int_std]
            a22=ary_blk_v[6::int_std]
            a23=ary_blk_v[7::int_std]
            a24=ary_blk_v[8::int_std]
            a25=ary_blk_v[9::int_std]
            a31=ary_blk_v[10::int_std]
            a32=ary_blk_v[11::int_std]
            a33=ary_blk_v[12::int_std]
            a34=ary_blk_v[13::int_std]
            a35=ary_blk_v[14::int_std]
            a41=ary_blk_v[15::int_std]
            a42=ary_blk_v[16::int_std]
            a43=ary_blk_v[17::int_std]
            a44=ary_blk_v[18::int_std]
            a45=ary_blk_v[19::int_std]
            a51=ary_blk_v[20::int_std]
            a52=ary_blk_v[21::int_std]
            a53=ary_blk_v[22::int_std]
            a54=ary_blk_v[23::int_std]
            a55=ary_blk_v[24::int_std]
            detA=(a11*a22*a33*a44*a55 - a11*a22*a33*a45*a54 - a11*a22*a34*a43*a55 + 
                  a11*a22*a34*a45*a53 + a11*a22*a35*a43*a54 - a11*a22*a35*a44*a53 - 
                  a11*a23*a32*a44*a55 + a11*a23*a32*a45*a54 + a11*a23*a34*a42*a55 - 
                  a11*a23*a34*a45*a52 - a11*a23*a35*a42*a54 + a11*a23*a35*a44*a52 + 
                  a11*a24*a32*a43*a55 - a11*a24*a32*a45*a53 - a11*a24*a33*a42*a55 + 
                  a11*a24*a33*a45*a52 + a11*a24*a35*a42*a53 - a11*a24*a35*a43*a52 - 
                  a11*a25*a32*a43*a54 + a11*a25*a32*a44*a53 + a11*a25*a33*a42*a54 - 
                  a11*a25*a33*a44*a52 - a11*a25*a34*a42*a53 + a11*a25*a34*a43*a52 - 
                  a12*a21*a33*a44*a55 + a12*a21*a33*a45*a54 + a12*a21*a34*a43*a55 - 
                  a12*a21*a34*a45*a53 - a12*a21*a35*a43*a54 + a12*a21*a35*a44*a53 + 
                  a12*a23*a31*a44*a55 - a12*a23*a31*a45*a54 - a12*a23*a34*a41*a55 + 
                  a12*a23*a34*a45*a51 + a12*a23*a35*a41*a54 - a12*a23*a35*a44*a51 - 
                  a12*a24*a31*a43*a55 + a12*a24*a31*a45*a53 + a12*a24*a33*a41*a55 - 
                  a12*a24*a33*a45*a51 - a12*a24*a35*a41*a53 + a12*a24*a35*a43*a51 + 
                  a12*a25*a31*a43*a54 - a12*a25*a31*a44*a53 - a12*a25*a33*a41*a54 + 
                  a12*a25*a33*a44*a51 + a12*a25*a34*a41*a53 - a12*a25*a34*a43*a51 + 
                  a13*a21*a32*a44*a55 - a13*a21*a32*a45*a54 - a13*a21*a34*a42*a55 + 
                  a13*a21*a34*a45*a52 + a13*a21*a35*a42*a54 - a13*a21*a35*a44*a52 - 
                  a13*a22*a31*a44*a55 + a13*a22*a31*a45*a54 + a13*a22*a34*a41*a55 - 
                  a13*a22*a34*a45*a51 - a13*a22*a35*a41*a54 + a13*a22*a35*a44*a51 + 
                  a13*a24*a31*a42*a55 - a13*a24*a31*a45*a52 - a13*a24*a32*a41*a55 + 
                  a13*a24*a32*a45*a51 + a13*a24*a35*a41*a52 - a13*a24*a35*a42*a51 - 
                  a13*a25*a31*a42*a54 + a13*a25*a31*a44*a52 + a13*a25*a32*a41*a54 - 
                  a13*a25*a32*a44*a51 - a13*a25*a34*a41*a52 + a13*a25*a34*a42*a51 - 
                  a14*a21*a32*a43*a55 + a14*a21*a32*a45*a53 + a14*a21*a33*a42*a55 - 
                  a14*a21*a33*a45*a52 - a14*a21*a35*a42*a53 + a14*a21*a35*a43*a52 + 
                  a14*a22*a31*a43*a55 - a14*a22*a31*a45*a53 - a14*a22*a33*a41*a55 + 
                  a14*a22*a33*a45*a51 + a14*a22*a35*a41*a53 - a14*a22*a35*a43*a51 - 
                  a14*a23*a31*a42*a55 + a14*a23*a31*a45*a52 + a14*a23*a32*a41*a55 - 
                  a14*a23*a32*a45*a51 - a14*a23*a35*a41*a52 + a14*a23*a35*a42*a51 + 
                  a14*a25*a31*a42*a53 - a14*a25*a31*a43*a52 - a14*a25*a32*a41*a53 + 
                  a14*a25*a32*a43*a51 + a14*a25*a33*a41*a52 - a14*a25*a33*a42*a51 + 
                  a15*a21*a32*a43*a54 - a15*a21*a32*a44*a53 - a15*a21*a33*a42*a54 + 
                  a15*a21*a33*a44*a52 + a15*a21*a34*a42*a53 - a15*a21*a34*a43*a52 - 
                  a15*a22*a31*a43*a54 + a15*a22*a31*a44*a53 + a15*a22*a33*a41*a54 - 
                  a15*a22*a33*a44*a51 - a15*a22*a34*a41*a53 + a15*a22*a34*a43*a51 + 
                  a15*a23*a31*a42*a54 - a15*a23*a31*a44*a52 - a15*a23*a32*a41*a54 + 
                  a15*a23*a32*a44*a51 + a15*a23*a34*a41*a52 - a15*a23*a34*a42*a51 - 
                  a15*a24*a31*a42*a53 + a15*a24*a31*a43*a52 + a15*a24*a32*a41*a53 - 
                  a15*a24*a32*a43*a51 - a15*a24*a33*a41*a52 + a15*a24*a33*a42*a51)
            b11=(a22*a33*a44*a55 - a22*a33*a45*a54 - a22*a34*a43*a55 + a22*a34*a45*a53 + 
                 a22*a35*a43*a54 - a22*a35*a44*a53 - a23*a32*a44*a55 + a23*a32*a45*a54 + 
                 a23*a34*a42*a55 - a23*a34*a45*a52 - a23*a35*a42*a54 + a23*a35*a44*a52 + 
                 a24*a32*a43*a55 - a24*a32*a45*a53 - a24*a33*a42*a55 + a24*a33*a45*a52 + 
                 a24*a35*a42*a53 - a24*a35*a43*a52 - a25*a32*a43*a54 + a25*a32*a44*a53 + 
                 a25*a33*a42*a54 - a25*a33*a44*a52 - a25*a34*a42*a53 + a25*a34*a43*a52)
            b12=-(a12*a33*a44*a55 - a12*a33*a45*a54 - a12*a34*a43*a55 + a12*a34*a45*a53 +
                  a12*a35*a43*a54 - a12*a35*a44*a53 - a13*a32*a44*a55 + a13*a32*a45*a54 + 
                  a13*a34*a42*a55 - a13*a34*a45*a52 - a13*a35*a42*a54 + a13*a35*a44*a52 +
                  a14*a32*a43*a55 - a14*a32*a45*a53 - a14*a33*a42*a55 + a14*a33*a45*a52 +
                  a14*a35*a42*a53 - a14*a35*a43*a52 - a15*a32*a43*a54 + a15*a32*a44*a53 +
                  a15*a33*a42*a54 - a15*a33*a44*a52 - a15*a34*a42*a53 + a15*a34*a43*a52)
            b13=(a12*a23*a44*a55 - a12*a23*a45*a54 - a12*a24*a43*a55 + a12*a24*a45*a53 + 
                 a12*a25*a43*a54 - a12*a25*a44*a53 - a13*a22*a44*a55 + a13*a22*a45*a54 + 
                 a13*a24*a42*a55 - a13*a24*a45*a52 - a13*a25*a42*a54 + a13*a25*a44*a52 + 
                 a14*a22*a43*a55 - a14*a22*a45*a53 - a14*a23*a42*a55 + a14*a23*a45*a52 + 
                 a14*a25*a42*a53 - a14*a25*a43*a52 - a15*a22*a43*a54 + a15*a22*a44*a53 + 
                 a15*a23*a42*a54 - a15*a23*a44*a52 - a15*a24*a42*a53 + a15*a24*a43*a52)
            b14=-(a12*a23*a34*a55 - a12*a23*a35*a54 - a12*a24*a33*a55 + a12*a24*a35*a53 +
                  a12*a25*a33*a54 - a12*a25*a34*a53 - a13*a22*a34*a55 + a13*a22*a35*a54 +
                  a13*a24*a32*a55 - a13*a24*a35*a52 - a13*a25*a32*a54 + a13*a25*a34*a52 +
                  a14*a22*a33*a55 - a14*a22*a35*a53 - a14*a23*a32*a55 + a14*a23*a35*a52 +
                  a14*a25*a32*a53 - a14*a25*a33*a52 - a15*a22*a33*a54 + a15*a22*a34*a53 +
                  a15*a23*a32*a54 - a15*a23*a34*a52 - a15*a24*a32*a53 + a15*a24*a33*a52)
            b15=(a12*a23*a34*a45 - a12*a23*a35*a44 - a12*a24*a33*a45 + a12*a24*a35*a43 + 
                 a12*a25*a33*a44 - a12*a25*a34*a43 - a13*a22*a34*a45 + a13*a22*a35*a44 + 
                 a13*a24*a32*a45 - a13*a24*a35*a42 - a13*a25*a32*a44 + a13*a25*a34*a42 + 
                 a14*a22*a33*a45 - a14*a22*a35*a43 - a14*a23*a32*a45 + a14*a23*a35*a42 + 
                 a14*a25*a32*a43 - a14*a25*a33*a42 - a15*a22*a33*a44 + a15*a22*a34*a43 + 
                 a15*a23*a32*a44 - a15*a23*a34*a42 - a15*a24*a32*a43 + a15*a24*a33*a42)
            b21=-(a21*a33*a44*a55 - a21*a33*a45*a54 - a21*a34*a43*a55 + a21*a34*a45*a53 +
                  a21*a35*a43*a54 - a21*a35*a44*a53 - a23*a31*a44*a55 + a23*a31*a45*a54 +
                  a23*a34*a41*a55 - a23*a34*a45*a51 - a23*a35*a41*a54 + a23*a35*a44*a51 +
                  a24*a31*a43*a55 - a24*a31*a45*a53 - a24*a33*a41*a55 + a24*a33*a45*a51 +
                  a24*a35*a41*a53 - a24*a35*a43*a51 - a25*a31*a43*a54 + a25*a31*a44*a53 +
                  a25*a33*a41*a54 - a25*a33*a44*a51 - a25*a34*a41*a53 + a25*a34*a43*a51)
            b22=(a11*a33*a44*a55 - a11*a33*a45*a54 - a11*a34*a43*a55 + a11*a34*a45*a53 + 
                 a11*a35*a43*a54 - a11*a35*a44*a53 - a13*a31*a44*a55 + a13*a31*a45*a54 +
                 a13*a34*a41*a55 - a13*a34*a45*a51 - a13*a35*a41*a54 + a13*a35*a44*a51 +
                 a14*a31*a43*a55 - a14*a31*a45*a53 - a14*a33*a41*a55 + a14*a33*a45*a51 +
                 a14*a35*a41*a53 - a14*a35*a43*a51 - a15*a31*a43*a54 + a15*a31*a44*a53 +
                 a15*a33*a41*a54 - a15*a33*a44*a51 - a15*a34*a41*a53 + a15*a34*a43*a51)
            b23=-(a11*a23*a44*a55 - a11*a23*a45*a54 - a11*a24*a43*a55 + a11*a24*a45*a53 +
                  a11*a25*a43*a54 - a11*a25*a44*a53 - a13*a21*a44*a55 + a13*a21*a45*a54 +
                  a13*a24*a41*a55 - a13*a24*a45*a51 - a13*a25*a41*a54 + a13*a25*a44*a51 +
                  a14*a21*a43*a55 - a14*a21*a45*a53 - a14*a23*a41*a55 + a14*a23*a45*a51 +
                  a14*a25*a41*a53 - a14*a25*a43*a51 - a15*a21*a43*a54 + a15*a21*a44*a53 +
                  a15*a23*a41*a54 - a15*a23*a44*a51 - a15*a24*a41*a53 + a15*a24*a43*a51)
            b24=(a11*a23*a34*a55 - a11*a23*a35*a54 - a11*a24*a33*a55 + a11*a24*a35*a53 +
                 a11*a25*a33*a54 - a11*a25*a34*a53 - a13*a21*a34*a55 + a13*a21*a35*a54 +
                 a13*a24*a31*a55 - a13*a24*a35*a51 - a13*a25*a31*a54 + a13*a25*a34*a51 +
                 a14*a21*a33*a55 - a14*a21*a35*a53 - a14*a23*a31*a55 + a14*a23*a35*a51 +
                 a14*a25*a31*a53 - a14*a25*a33*a51 - a15*a21*a33*a54 + a15*a21*a34*a53 +
                 a15*a23*a31*a54 - a15*a23*a34*a51 - a15*a24*a31*a53 + a15*a24*a33*a51)
            b25=-(a11*a23*a34*a45 - a11*a23*a35*a44 - a11*a24*a33*a45 + a11*a24*a35*a43 +
                  a11*a25*a33*a44 - a11*a25*a34*a43 - a13*a21*a34*a45 + a13*a21*a35*a44 +
                  a13*a24*a31*a45 - a13*a24*a35*a41 - a13*a25*a31*a44 + a13*a25*a34*a41 +
                  a14*a21*a33*a45 - a14*a21*a35*a43 - a14*a23*a31*a45 + a14*a23*a35*a41 +
                  a14*a25*a31*a43 - a14*a25*a33*a41 - a15*a21*a33*a44 + a15*a21*a34*a43 +
                  a15*a23*a31*a44 - a15*a23*a34*a41 - a15*a24*a31*a43 + a15*a24*a33*a41)
            b31=(a21*a32*a44*a55 - a21*a32*a45*a54 - a21*a34*a42*a55 + a21*a34*a45*a52 +
                 a21*a35*a42*a54 - a21*a35*a44*a52 - a22*a31*a44*a55 + a22*a31*a45*a54 +
                 a22*a34*a41*a55 - a22*a34*a45*a51 - a22*a35*a41*a54 + a22*a35*a44*a51 +
                 a24*a31*a42*a55 - a24*a31*a45*a52 - a24*a32*a41*a55 + a24*a32*a45*a51 +
                 a24*a35*a41*a52 - a24*a35*a42*a51 - a25*a31*a42*a54 + a25*a31*a44*a52 +
                 a25*a32*a41*a54 - a25*a32*a44*a51 - a25*a34*a41*a52 + a25*a34*a42*a51)
            b32=-(a11*a32*a44*a55 - a11*a32*a45*a54 - a11*a34*a42*a55 + a11*a34*a45*a52 +
                  a11*a35*a42*a54 - a11*a35*a44*a52 - a12*a31*a44*a55 + a12*a31*a45*a54 +
                  a12*a34*a41*a55 - a12*a34*a45*a51 - a12*a35*a41*a54 + a12*a35*a44*a51 +
                  a14*a31*a42*a55 - a14*a31*a45*a52 - a14*a32*a41*a55 + a14*a32*a45*a51 +
                  a14*a35*a41*a52 - a14*a35*a42*a51 - a15*a31*a42*a54 + a15*a31*a44*a52 +
                  a15*a32*a41*a54 - a15*a32*a44*a51 - a15*a34*a41*a52 + a15*a34*a42*a51)
            b33=(a11*a22*a44*a55 - a11*a22*a45*a54 - a11*a24*a42*a55 + a11*a24*a45*a52 +
                 a11*a25*a42*a54 - a11*a25*a44*a52 - a12*a21*a44*a55 + a12*a21*a45*a54 +
                 a12*a24*a41*a55 - a12*a24*a45*a51 - a12*a25*a41*a54 + a12*a25*a44*a51 +
                 a14*a21*a42*a55 - a14*a21*a45*a52 - a14*a22*a41*a55 + a14*a22*a45*a51 +
                 a14*a25*a41*a52 - a14*a25*a42*a51 - a15*a21*a42*a54 + a15*a21*a44*a52 +
                 a15*a22*a41*a54 - a15*a22*a44*a51 - a15*a24*a41*a52 + a15*a24*a42*a51)
            b34=-(a11*a22*a34*a55 - a11*a22*a35*a54 - a11*a24*a32*a55 + a11*a24*a35*a52 +
                  a11*a25*a32*a54 - a11*a25*a34*a52 - a12*a21*a34*a55 + a12*a21*a35*a54 +
                  a12*a24*a31*a55 - a12*a24*a35*a51 - a12*a25*a31*a54 + a12*a25*a34*a51 +
                  a14*a21*a32*a55 - a14*a21*a35*a52 - a14*a22*a31*a55 + a14*a22*a35*a51 +
                  a14*a25*a31*a52 - a14*a25*a32*a51 - a15*a21*a32*a54 + a15*a21*a34*a52 +
                  a15*a22*a31*a54 - a15*a22*a34*a51 - a15*a24*a31*a52 + a15*a24*a32*a51)
            b35=(a11*a22*a34*a45 - a11*a22*a35*a44 - a11*a24*a32*a45 + a11*a24*a35*a42 +
                 a11*a25*a32*a44 - a11*a25*a34*a42 - a12*a21*a34*a45 + a12*a21*a35*a44 +
                 a12*a24*a31*a45 - a12*a24*a35*a41 - a12*a25*a31*a44 + a12*a25*a34*a41 +
                 a14*a21*a32*a45 - a14*a21*a35*a42 - a14*a22*a31*a45 + a14*a22*a35*a41 +
                 a14*a25*a31*a42 - a14*a25*a32*a41 - a15*a21*a32*a44 + a15*a21*a34*a42 +
                 a15*a22*a31*a44 - a15*a22*a34*a41 - a15*a24*a31*a42 + a15*a24*a32*a41)
            b41=-(a21*a32*a43*a55 - a21*a32*a45*a53 - a21*a33*a42*a55 + a21*a33*a45*a52 +
                  a21*a35*a42*a53 - a21*a35*a43*a52 - a22*a31*a43*a55 + a22*a31*a45*a53 +
                  a22*a33*a41*a55 - a22*a33*a45*a51 - a22*a35*a41*a53 + a22*a35*a43*a51 +
                  a23*a31*a42*a55 - a23*a31*a45*a52 - a23*a32*a41*a55 + a23*a32*a45*a51 +
                  a23*a35*a41*a52 - a23*a35*a42*a51 - a25*a31*a42*a53 + a25*a31*a43*a52 +
                  a25*a32*a41*a53 - a25*a32*a43*a51 - a25*a33*a41*a52 + a25*a33*a42*a51)
            b42=(a11*a32*a43*a55 - a11*a32*a45*a53 - a11*a33*a42*a55 + a11*a33*a45*a52 +
                 a11*a35*a42*a53 - a11*a35*a43*a52 - a12*a31*a43*a55 + a12*a31*a45*a53 +
                 a12*a33*a41*a55 - a12*a33*a45*a51 - a12*a35*a41*a53 + a12*a35*a43*a51 +
                 a13*a31*a42*a55 - a13*a31*a45*a52 - a13*a32*a41*a55 + a13*a32*a45*a51 +
                 a13*a35*a41*a52 - a13*a35*a42*a51 - a15*a31*a42*a53 + a15*a31*a43*a52 +
                 a15*a32*a41*a53 - a15*a32*a43*a51 - a15*a33*a41*a52 + a15*a33*a42*a51)
            b43=-(a11*a22*a43*a55 - a11*a22*a45*a53 - a11*a23*a42*a55 + a11*a23*a45*a52 +
                  a11*a25*a42*a53 - a11*a25*a43*a52 - a12*a21*a43*a55 + a12*a21*a45*a53 +
                  a12*a23*a41*a55 - a12*a23*a45*a51 - a12*a25*a41*a53 + a12*a25*a43*a51 +
                  a13*a21*a42*a55 - a13*a21*a45*a52 - a13*a22*a41*a55 + a13*a22*a45*a51 +
                  a13*a25*a41*a52 - a13*a25*a42*a51 - a15*a21*a42*a53 + a15*a21*a43*a52 +
                  a15*a22*a41*a53 - a15*a22*a43*a51 - a15*a23*a41*a52 + a15*a23*a42*a51)
            b44=(a11*a22*a33*a55 - a11*a22*a35*a53 - a11*a23*a32*a55 + a11*a23*a35*a52 +
                 a11*a25*a32*a53 - a11*a25*a33*a52 - a12*a21*a33*a55 + a12*a21*a35*a53 +
                 a12*a23*a31*a55 - a12*a23*a35*a51 - a12*a25*a31*a53 + a12*a25*a33*a51 + 
                 a13*a21*a32*a55 - a13*a21*a35*a52 - a13*a22*a31*a55 + a13*a22*a35*a51 +
                 a13*a25*a31*a52 - a13*a25*a32*a51 - a15*a21*a32*a53 + a15*a21*a33*a52 +
                 a15*a22*a31*a53 - a15*a22*a33*a51 - a15*a23*a31*a52 + a15*a23*a32*a51)
            b45=-(a11*a22*a33*a45 - a11*a22*a35*a43 - a11*a23*a32*a45 + a11*a23*a35*a42 +
                  a11*a25*a32*a43 - a11*a25*a33*a42 - a12*a21*a33*a45 + a12*a21*a35*a43 +
                  a12*a23*a31*a45 - a12*a23*a35*a41 - a12*a25*a31*a43 + a12*a25*a33*a41 +
                  a13*a21*a32*a45 - a13*a21*a35*a42 - a13*a22*a31*a45 + a13*a22*a35*a41 +
                  a13*a25*a31*a42 - a13*a25*a32*a41 - a15*a21*a32*a43 + a15*a21*a33*a42 +
                  a15*a22*a31*a43 - a15*a22*a33*a41 - a15*a23*a31*a42 + a15*a23*a32*a41)
            b51=(a21*a32*a43*a54 - a21*a32*a44*a53 - a21*a33*a42*a54 + a21*a33*a44*a52 +
                 a21*a34*a42*a53 - a21*a34*a43*a52 - a22*a31*a43*a54 + a22*a31*a44*a53 +
                 a22*a33*a41*a54 - a22*a33*a44*a51 - a22*a34*a41*a53 + a22*a34*a43*a51 +
                 a23*a31*a42*a54 - a23*a31*a44*a52 - a23*a32*a41*a54 + a23*a32*a44*a51 +
                 a23*a34*a41*a52 - a23*a34*a42*a51 - a24*a31*a42*a53 + a24*a31*a43*a52 +
                 a24*a32*a41*a53 - a24*a32*a43*a51 - a24*a33*a41*a52 + a24*a33*a42*a51)
            b52=-(a11*a32*a43*a54 - a11*a32*a44*a53 - a11*a33*a42*a54 + a11*a33*a44*a52 +
                  a11*a34*a42*a53 - a11*a34*a43*a52 - a12*a31*a43*a54 + a12*a31*a44*a53 +
                  a12*a33*a41*a54 - a12*a33*a44*a51 - a12*a34*a41*a53 + a12*a34*a43*a51 +
                  a13*a31*a42*a54 - a13*a31*a44*a52 - a13*a32*a41*a54 + a13*a32*a44*a51 +
                  a13*a34*a41*a52 - a13*a34*a42*a51 - a14*a31*a42*a53 + a14*a31*a43*a52 +
                  a14*a32*a41*a53 - a14*a32*a43*a51 - a14*a33*a41*a52 + a14*a33*a42*a51)
            b53=(a11*a22*a43*a54 - a11*a22*a44*a53 - a11*a23*a42*a54 + a11*a23*a44*a52 +
                 a11*a24*a42*a53 - a11*a24*a43*a52 - a12*a21*a43*a54 + a12*a21*a44*a53 +
                 a12*a23*a41*a54 - a12*a23*a44*a51 - a12*a24*a41*a53 + a12*a24*a43*a51 +
                 a13*a21*a42*a54 - a13*a21*a44*a52 - a13*a22*a41*a54 + a13*a22*a44*a51 +
                 a13*a24*a41*a52 - a13*a24*a42*a51 - a14*a21*a42*a53 + a14*a21*a43*a52 +
                 a14*a22*a41*a53 - a14*a22*a43*a51 - a14*a23*a41*a52 + a14*a23*a42*a51)
            b54=-(a11*a22*a33*a54 - a11*a22*a34*a53 - a11*a23*a32*a54 + a11*a23*a34*a52 +
                  a11*a24*a32*a53 - a11*a24*a33*a52 - a12*a21*a33*a54 + a12*a21*a34*a53 +
                  a12*a23*a31*a54 - a12*a23*a34*a51 - a12*a24*a31*a53 + a12*a24*a33*a51 +
                  a13*a21*a32*a54 - a13*a21*a34*a52 - a13*a22*a31*a54 + a13*a22*a34*a51 +
                  a13*a24*a31*a52 - a13*a24*a32*a51 - a14*a21*a32*a53 + a14*a21*a33*a52 +
                  a14*a22*a31*a53 - a14*a22*a33*a51 - a14*a23*a31*a52 + a14*a23*a32*a51)
            b55=(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 +
                 a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 +
                 a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 +
                 a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 +
                 a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 +
                 a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41)
            minor_mtx[0::int_std]=b11
            minor_mtx[1::int_std]=b12
            minor_mtx[2::int_std]=b13
            minor_mtx[3::int_std]=b14
            minor_mtx[4::int_std]=b15
            minor_mtx[5::int_std]=b21
            minor_mtx[6::int_std]=b22
            minor_mtx[7::int_std]=b23
            minor_mtx[8::int_std]=b24
            minor_mtx[9::int_std]=b25
            minor_mtx[10::int_std]=b31
            minor_mtx[11::int_std]=b32
            minor_mtx[12::int_std]=b33
            minor_mtx[13::int_std]=b34
            minor_mtx[14::int_std]=b35
            minor_mtx[15::int_std]=b41
            minor_mtx[16::int_std]=b42
            minor_mtx[17::int_std]=b43
            minor_mtx[18::int_std]=b44
            minor_mtx[19::int_std]=b45
            minor_mtx[20::int_std]=b51
            minor_mtx[21::int_std]=b52
            minor_mtx[22::int_std]=b53
            minor_mtx[23::int_std]=b54
            minor_mtx[24::int_std]=b55
            ary_blk_v=minor_mtx / np.repeat(detA,int_std)
            del minor_mtx
        else:
            #no quick way to invert other-sized blocks in one go (for now), so do them individually
            num_blocks=ary_blk_v.size/int_std
            #inversion and replacement
            ary_blk_v=np.hstack([inv(ary_blk_v[b_*int_std:
                                               (b_+1)*int_std].reshape(block_sz,block_sz)).flatten()
                                               for b_ in xrange(num_blocks)])
        #store elements corresponding to inverted blocks in new array
        new_csr_data[dict_bdiag[block_sz]]=ary_blk_v
    return csr_matrix((new_csr_data,(csr_rows,csr_cols)),shape=csr_bdiag.shape)
    
def mad(data, axis=None):
    return median(absolute(data - median(data, axis)), axis)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
