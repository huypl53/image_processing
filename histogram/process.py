#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : process.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 23.01.2021
# Last Modified Date: 24.01.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>
# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '../../')

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
if __name__ == "__main__":
    image = cv2.imread('../Fig0222(a)(face).tif', 0)
    # %%
    dist = get_distribution(image)
    # %%
    # cv2.imshow('Origin', image)
    # t = hist(dist)
    # hist_plt, a, b = plt.hist(np.ravel(image), 20)
    # hist_np = cv2.calcHist([image], [0], None, [20], ranges=[0,256])
    # plt.show()
    
    cv2.imshow('Origin', image)
    equaled_hist_img = equal_hist(image)
    cv2.imshow('Equal hist', equaled_hist_img)
    cv_eq_img = cv2.equalizeHist(image)
    cv2.imshow('Cv2 equl hist', cv_eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow('Origin', image)
    # t = hist(dist)
    # hist_plt, a, b = plt.hist(np.ravel(image), 20)
    # hist_np = cv2.calcHist([image], [0], None, [20], ranges=[0,256])
    # plt.show()
    
    cv2.imshow('Origin', image)
    equaled_hist_img = equal_hist(image)
    cv2.imshow('Equal hist', equaled_hist_img)
    cv_eq_img = cv2.equalizeHist(image)
    cv2.imshow('Cv2 equl hist', cv_eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # %%
    plt.subplot(1,2,1)
    plt.hist(np.ravel(cv_eq_img), bins=20)
    plt.subplot(1,2,2)
    plt.hist(np.ravel(equaled_hist_img), bins=20)
    plt.show()

