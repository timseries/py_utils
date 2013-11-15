#!/usr/bin/python -tt
"""
a collection of matrix utilities
"""
import numpy as np

def nd_impulse(ary_size):
    ary_impulse = np.zeros(ary_size)
    ary_impulse[tuple(np.array(ary_impulse.shape)/2)] = 1
    return ary_impulse
