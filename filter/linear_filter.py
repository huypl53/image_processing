#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear_filter.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 29.01.2021
# Last Modified Date: 04.02.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>

# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os
from skimage.exposure import rescale_intensity
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
    assert kernel.shape[0]%2==1 and kernel.shape[1]%2==1
    return kernel[::-1, ::-1, ...]

def conv2d(src_image: np.ndarray, kernel: np.ndarray):

    assert src_image.any() and kernel.any()
    src_image = src_image.copy()
    src_type = src_image.dtype
    # src_image = src_image.astype(np.float)
    # cv2.imshow('Src', src_image)
    krn_h, krn_w = kernel.shape[:2]
    if krn_h%2+krn_w%2!=2:
        print('Warning: kernel dims not odd')
    mask = rotate_kernel(kernel)
    # mask = (kernel)
    # print(src_image.shape)
    # print(np.array(src_image.shape[:2])+2*np.array([krn_h//2, krn_w//2]))
    pad_zeros = np.zeros(np.array(src_image.shape[:2])+2*np.array([krn_h//2, krn_w//2]), dtype=src_image.dtype)
    win_w = krn_w//2
    win_h = krn_h//2
    pad_zeros[win_h:src_image.shape[0]+win_h, win_w:src_image.shape[1]+win_w]  = src_image
    ret_image = np.zeros_like(src_image, dtype=np.float)
    for i in range(src_image.shape[0]):
        for j in range(src_image.shape[1]):
            ret_image[i, j] = np.sum(pad_zeros[ i:i+krn_h, j:j+krn_w ]*mask)
    i_min = np.min(ret_image)
    i_max = np.max(ret_image)
    ret_image = np.clip(ret_image, i_min, i_max)
    if i_min==i_max:
        return ret_image-i_min
    else:
        return ((ret_image-i_min)/(i_max-i_min)*(L-1)).astype(src_type)
    # return rescale_intensity(ret_image, in_range=(0, 255))

# %%
if __name__ == "__main__":
    image = cv2.imread(os.path.join(CURRENT_DIR, '../Fig0222(a)(face).tif'), 0)
    low_pass_krnl = np.array([[ 0, 1/8, 0 ],
                             [1/8, 5/8, 1/8],
                             [0, 1/8, 0]])
    high_pass_krnl = np.array([[-1/9, -1/9, -1/9],
                              [-1/9, 8/9, -1/9],
                              [-1/9, -1/9, -1/9]])
    # edge_krnl = np.array([[0, 1, 1, 1, 1],
    #                      [-1, 0, 1, 1, 1],
    #                      [-1, -1, 0, 1, 1],
    #                      [-1, -1, -1, 0, 1],
    #                      [-1, -1, -1, -1, 0]])
    edge_krnl = np.array([[-2, 0, 2],
                          [-2, 0, 2],
                          [-2, 0, 2]])
# %%
    low_pass_image = conv2d(image, low_pass_krnl)
    edge_detect_image = conv2d(image, edge_krnl)
    high_pass_image = conv2d(image, high_pass_krnl)
    cv2_edge_image = cv2.filter2D(image, -1, edge_krnl)
    cv2.imshow('Origin', image)
    cv2.imshow('Low pass', low_pass_image)
    cv2.imshow('High passed', high_pass_image)
    cv2.imshow('Edege detect', edge_detect_image)
    cv2.imshow('Cv2 edge krnl', cv2_edge_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
