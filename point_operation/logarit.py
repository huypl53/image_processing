#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : logarit.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 22.01.2021
# Last Modified Date: 22.01.2021
# Last Modified By  : phamlehuy53 <unknownsol98@gmail>
# %%
import numpy as np
import cv2
import sys
sys.path.insert(1, '../../')

# %%
def log(image: np.ndarray = None):
    # For non-devided by zero
    image = np.log(image+5e-1)
    return np.uint8(image)

def inverse_log(image: np.ndarray = None):
    image = np.exp(image)
    return np.uint8(image)
# %%
if __name__ == "__main__":
    image = cv2.imread('../Fig0222(a)(face).tif')
    # %%
    cv2.imshow('Origin', image)
    cv2.imshow('Log', log(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
