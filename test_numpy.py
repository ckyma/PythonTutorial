# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:08:19 2014

@author: phinary0
"""

###############################################################################
# Chapter 2. Numpy Basics

import numpy as np

###
# 2.1.1 Array types and conversions between types
###
#%%
z = np.arange(3, dtype=np.uint8)
z
x = z.astype(float)
x
z.dtype
x.dtype
d = np.dtype(np.uint8)
np.issubdtype(d, np.uint8)
np.issubdtype(d, int)
#%%