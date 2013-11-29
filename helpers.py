#!/usr/bin/python -tt
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