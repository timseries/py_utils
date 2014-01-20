#!/usr/bin/python -tt
from scipy.io import savemat,loadmat

def convertStr(s):
    """Convert string to int, output 'error' othwerise."""
    try:
        ret = int(s)
    except ValueError:
        #Try float.
        if s=='q':
            ret = 'q'
        else:    
            ret = 'error'
    return ret

def find_mount_point(path):
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path

def numpy_to_mat(ary_numpy_array, str_filepath, str_member='ary_numpy_array'):
    """
    Flatten a numpy array using column major ordering, then save to str_filepath.
    File member will always be 'ary_numpy_array'
    """
    data = {}
    data[str_member] = ary_numpy_array
    savemat(str_filepath,data)
