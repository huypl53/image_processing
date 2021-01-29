#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : logarit.py
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
def log(image: np.ndarray = None):
    # For non-devided by zero
    image = image.copy().astype(np.float)
    image = np.log(image+5e-1)

    return np.uint8(image)

def inverse_log(image: np.ndarray = None):
    image = image.copy().astype(np.uint32)
    m = np.max(image)
    if np.exp(m) > 255:
        print(f'WARNING: Pixel intensity is over 255!')
    image = np.exp(image)
    return np.uint8(image)
# %%
if __name__ == "__main__":
    image = cv2.imread(os.path.join(CURRENT_DIR, '../Fig0222(a)(face).tif'), 0)
    # %%
    cv2.imshow('Origin', image)
    cv2.imshow('Log', log(image))
    cv2.imshow('Inverse-log', inverse_log(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
