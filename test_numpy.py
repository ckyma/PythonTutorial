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

###
# 2.4.5 Indexing Multi-dimensional arrays
###
#%%
y = np.arange(35).reshape(5, 7)
#%%
#%%
y[np.array([0, 2, 4]), np.array([0, 1, 2])]
#%%

###
# 2.4.8 Structural indexing tools
###
#%%
x = np.arange(5)
x[:, np.newaxis]
x[np.newaxis, :]
x[:, np.newaxis] + x[np.newaxis, :]
#%%
#%%
z = np.arange(81).reshape(3, 3, 3, 3)
z
z[1, ..., 2]
# equivalent to
z[1, :, :, 2]
#%%
#%%
x = np.arange(0, 50, 10)
x
x[np.array([1, 1, 3, 1])] += 1
x
#%%

###
# 2.4.10 Dealing with variable numbers of indices within programs
###
#%%
indices = (1, 1, 1, slice(0, 2))
z[indices]
# equivalent to
z[1, 1, 1, 0:2]

indices = (1, Ellipsis, 1)
z[indices]
# equivalent to
z[1, :, :, 1]
#%%

###
# 2.5.1 General Broadcasting Rules

###
#%%
x = np.arange(4)
xx = x.reshape(4, 1)
y = np.ones(5)
xx + y
#%%

###
# 2.6.1 Introduction to byte ordering and ndarrays
###
#%%
big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
big_end_str
#%%