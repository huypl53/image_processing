#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear_filter.py
# Author            : phamlehuy53 <unknownsol98@gmail>
# Date              : 24.01.2021
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
def conv2d(image: np.ndarray, kernel: np.ndarray):
    # check if kernel's dim is odd
    if type(kernel) != np.ndarray:
        kernel = np.array(kernel)
    assert list(filter(lambda  x: x%2==1, kernel.shape[:2]))
    img = image.astype(np.float32)
    def rotate180_kernel(kernel: np.ndarray):
        kernel = kernel.copy()
        kernel = kernel[::-1, ::-1]
        return kernel

    rotated_kernel = rotate180_kernel(kernel)
    res_img = np.zeros_like(img)
    krnl_a, krnl_b = kernel.shape[0]//2, kernel.shape[1]//2
    h, w =img.shape[:2]
    padded_img = np.zeros( shape=(h+2*krnl_a, w+2*krnl_b, *img.shape[2:]) )
    padded_img[krnl_a:krnl_a+h, krnl_b:krnl_b+w, ...] = img
    for i in range(h):
        for j in range(w):
            res_img[i,j] = np.sum(padded_img[i:i+2*krnl_a+1, j:j+2*krnl_b+1]*rotated_kernel)

    return res_img.astype(image.dtype)

    

# %%
if __name__ == "__main__":
    image = cv2.imread('../Fig0222(a)(face).tif', 0)
# %%
    cv2.imshow('Origin', image)
    filtered_image = conv2d(image, kernel= [ [-1/9, -1/9, -1/9],
                                            [-1/9, 8/9, -1/9],
                                            [-1/9, -1/9, -1/9]])
    cv2.imshow('Edged', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
