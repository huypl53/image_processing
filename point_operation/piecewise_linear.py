#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : piecewise_linear.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 22.01.2021
# Last Modified Date: 29.01.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>
# %%
import numpy as np
import cv2
import os
import sys
CURRENT_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'
if CURRENT_DIR == '.':
        print('Warning: Program not run by source script!')
sys.path.insert(1, os.path.join( CURRENT_DIR, '../../'))

import image_processing
L = image_processing.L
# %%

def scale(image: np.ndarray = None,
          umin: int = 2,
          umax: int = 10,
          L:int = 256):
    image = (L-1)*(image-umin)/(umax-umin)
    return np.uint8(image)


def threshold(image: np.ndarray, v0:int, u0:int):
    img = image.copy()
    img[img>u0] = v0
    img[img<=u0] = 0
    return img

def stretch_constrast(image: np.ndarray,
                   s_u: [],
                   ):
    image = image.copy()
    image = np.uint16(image)
    v0 = 0 # max current v
    for i in range(1, len(s_u)):
        s = s_u[i][0]
        u = s_u[i][1]
        image[(image>s_u[i-1][1]) & (image<u)] = v0 + \
        (image[(image>s_u[i-1][1]) & (image<u)]-s_u[i-1][1])*s
        v0 += (u-s_u[i-1][1])*s
    image[image>L-2] = L-2
    return np.uint8(image)
# %%
if __name__ == "__main__":
    image = cv2.imread(os.path.join(CURRENT_DIR, '../Fig0222(a)(face).tif'), 0)
    # %%
    cv2.imshow('Origin', image)
    # cv2.imshow('Stretched',
    #            stretch_constrast(image, [[0, 10], [1.5, 50],  [1.75, 255]]))
    cv2.imshow('Stretched',
               stretch_constrast(image, [[0, 0], [1.5, 0],  [1.75, 222]]))
    cv2.imshow('Thresh', threshold(image,225, 125  ))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

