#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : powerlaw.py
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
# The general power-law
# transformation is given by
# v = c × u^gamma,
# where c = (L-1)^(1-gamma) .
def advanced_power(image: np.ndarray = None,
                   gamma: int = 1,
                   L: int = 256):
    c = (L-1)**(1-gamma)
    image = c * np.power(image, gamma)
    return np.uint8(image)

def power(image: np.ndarray = None,
          lv: int = 2):
    # For non-devided by zero
    image = np.power(image, lv)
    return np.uint8(image)

def root(image: np.ndarray = None,
         lv: int=2):
    image = np.power(image, 1./lv)
    return np.uint8(image)
# %%
if __name__ == "__main__":
    image = cv2.imread('../Fig0222(a)(face).tif')
    # %%
    cv2.imshow('Origin', image)
    cv2.imshow('Power', power(image, 1.2))
    cv2.imshow('Root', root(image, 1.2))
    gamma = 0.5
    cv2.imshow('Power with gamma {}'.format(gamma), advanced_power(image, gamma))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

