#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : process.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 23.01.2021
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
def get_distribution(image: np.ndarray):
    gray_dist = np.zeros( shape=(L, ), dtype=np.uint32 )
    flatten_img = np.ravel(image)
    for i in flatten_img:
        gray_dist[i] += 1
    return gray_dist

def hist(distribution: [] = None, bins: int = 20, y_step = None):
    distribution = np.array(distribution)
    l = len(distribution)
    step = l//bins
    # start_index = (l-step*bins)//2
    start_index = 0
    grouped_dist = { start_index+i*step: np.sum(distribution[start_index+step*(i-1): min(start_index+step*i, l)]) for i in range(1, bins+1) }
    # grouped_dist[start_index+1*step] += np.sum(distribution[:start_index])
    grouped_dist[start_index+bins*step] += np.sum(distribution[start_index+step*bins:])
    grouped_dist[l] = grouped_dist.pop(start_index+bins*step)
    plt.bar(grouped_dist.keys(), grouped_dist.values(), width=l/bins*1.00 )
    plt.show()
    return grouped_dist

def equal_hist(image: np.ndarray):
    img = image.copy()
    dist = get_distribution(image)
    # cumulative image histogram
    cuml = [dist[0]]
    for i in range(1, len(dist)):
        cuml.append(cuml[i-1]+dist[i])
    # for gray image only
    h, w = img.shape[:2]
    N = h*w
    for i in range(h):
        for j in range(w):
            img[i, j] = np.uint8((L-1)*cuml[img[i, j]]/N)
    return img
# %%

def match_hist(src_image: np.ndarray, dst_image: np.ndarray):
    """
    Recap - Histogram equalization method:
        T(j) = (L-1)*CDF(j)

        where j is the intensity level 
            CDF - cumulative distribution function ( the probility of random 
            choosen pixel having the gray level from 0 to j )

    The idea is:
            hist(src_image) --f()--> normalized_hist
            hist(dst_image) --g()--> normalized_hist( same as the above one)
        
        so f().inverse_g() is what we need
    """
    assert len(src_image.shape) < 3
    assert len(dst_image.shape) < 3

    s_ravel = src_image.ravel()
    d_ravel = dst_image.ravel()

    # Get the set of bins, their indices in origin array and their
    # quanties correspondingly
    s_bins, s_bin_idx, s_bin_cnt = np.unique(s_ravel,
                                             return_inverse=True,
                                             return_counts=True)

    d_bins, d_bin_idx, d_bin_cnt = np.unique(d_ravel,
                                             return_inverse=True,
                                             return_counts=True)

    # CDF-ed values
    s_cuml = np.cumsum(s_bin_cnt).astype(np.float)
    s_cdf = s_cuml/s_cuml[-1]
    d_cuml = np.cumsum(d_bin_cnt).astype(np.float)
    d_cdf = d_cuml/d_cuml[-1]

    # Parse into histogram range
    # here boths are L-1 = 255
    s_equa = s_cdf*(L-1)
    d_equa = d_cdf*(L-1)

    # inverse_g(d_equa) -> d_bins
    s_interp= np.interp(s_equa, d_equa, d_bins)

    res = s_interp[s_bin_idx].reshape(src_image.shape).astype(np.uint8)
    return res
# %%
if __name__ == "__main__":
    image = cv2.imread(os.path.join(CURRENT_DIR, '../Fig0222(a)(face).tif'), 0)
    hist_match_image = cv2.imread(os.path.join(CURRENT_DIR, '../download-(58).jpeg'), cv2.IMREAD_GRAYSCALE)
    hist_match_image = cv2.resize( hist_match_image, tuple(np.array(hist_match_image.shape[2::-1])//7) )
    # %%
    dist = get_distribution(image)
    
    # %%
    cv2.imshow('Origin', image)
    equaled_hist_img = equal_hist(image)
    cv2.imshow('Equal hist', equaled_hist_img)
    cv_eq_img = cv2.equalizeHist(image)
    cv2.imshow('Cv2 equl hist', cv_eq_img)

    cv2.imshow('Target image for histogram matching', hist_match_image)
    hist_matched = match_hist(image, hist_match_image)
    cv2.imshow('Image with matched histogram', hist_matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # %%
    plt.subplot(1,2,1)
    plt.hist(np.ravel(cv_eq_img), bins=20)
    plt.title('Equalize histogram by cv2')
    plt.subplot(1,2,2)
    plt.hist(np.ravel(equaled_hist_img), bins=20)
    plt.title('Equalize histogram manually')
    plt.show()

