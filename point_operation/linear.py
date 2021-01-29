#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 22.01.2021
# Last Modified Date: 29.01.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>
# %%
import numpy as np
import cv2
import sys
import os

CURRENT_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'
if CURRENT_DIR == '.':
        print('Warning: Program not run by source script!')
sys.path.insert(1, os.path.join( CURRENT_DIR, '../../'))

# %%
def negative(image: np.ndarray = None):
    if not image.any():
        print('Check input image!')
        return None
    image = 255 - image
    return image

# %%
def identity(image: np.ndarray = None):
    if not image.any():
        print('Check the image')
        return
    return image

# %%
if __name__ == "__main__":
    image = cv2.imread(os.path.join(CURRENT_DIR, '../Fig0222(a)(face).tif'), 0)
    # img.shape
    # %% [Negative test]
    cv2.imshow('Origin', image)
    cv2.imshow('Negatived', negative(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


