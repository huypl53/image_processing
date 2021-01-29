#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear_filter.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 29.01.2021
# Last Modified Date: 29.01.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>

# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os

CURRENT_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'
if CURRENT_DIR == '.':
        print('Warning: Program not run by source script!')
sys.path.insert(1, os.path.join( CURRENT_DIR, '../../'))
import image_processing
L = image_processing.L
# %%
def rotate_kernel(kernel: np.ndarray):
    """
    Rotate 180 deg
    Kernel's dim has to be odd

    Return: correlation mask
    """
    assert kernel.shape[0]%2==1 and kernel.shape[1]%==1
    return kernel[::-1, ::-1, ...]

def conv2d(src_image: np.ndarray, kernel: np.ndarray):


    krn_h, krn_w = kernel.shape[:2]
    if krn_h%2+krn_w%2!=2:
        print('Warning: kernel dims not odd')
    mask = rotate_kernel(kernel)

