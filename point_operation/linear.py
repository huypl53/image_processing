#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear.py
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
    img = cv2.imread('../Fig0222(a)(face).tif')
    img.shape
    # %% [Negative test]
    cv2.imshow('Origin', img)
    cv2.imshow('Negatived', negative(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


