import os
import re

import numpy as np
import pandas as pd


def gradien(array):
    ''' Calculate the gradien of an array on each point. Assuming that each 
    horizontal axis have the same step so the gradien is just the delta of its current value
    with the  next step'''
    grad = []
    for row in range(array.shape[0]-1):
        delta = array[row+1]- array[row]
        grad.append(delta)
    return grad

def find_turn_point(grad):
    '''Calculate the peak index of the target features array'''
    before_sign = -1
    pos_index = []
    for index, elem in enumerate(grad):
        if elem !=0:
            sign = elem/ abs(elem)
        else:
            sign = before_sign
        if sign < before_sign:
            pos_index.append(index)
        before_sign = sign

    return pos_index

def extract_mach_and_vf(file: str):
    '''Extract mach number and flutter speed index in file names'''
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))

