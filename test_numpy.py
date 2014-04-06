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
# 2.8.8 Slightly more realistic example - attribute added to existing array
###
#%%
class RealisticInfoArray(np.ndarray):
    
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, "info", None)

arr = np.arange(5)
obj = RealisticInfoArray(arr, info="information")
type(obj)
obj.info
v = obj[1:]
type(v)
v.info
#%%
# Without casting
#%%
class RealisticInfoArrayTest(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.ndarray.__new__(cls, input_array)
        obj.info = info
        return obj

arr = np.arange(5)
objTest = RealisticInfoArrayTest(arr, info="information")
type(objTest)
objTest.info
vTest = objTest[1:]
type(vTest)
vTest.info
#%%

###############################################################################
# Chapter 4. Miscellaneous

###
# 4.1 IEEE 754 Floating Point Special Values:
###
#%%
myarr = np.array([1., np.inf, np.nan, 3.])
# np.nan == np.nan doesn't work
np.where(myarr == np.nan)
np.where(np.isnan(myarr))
# True if not inf or nan
np.isfinite(myarr)
# map nan to 0, +/-inf to max/min float
np.nan_to_num(myarr)

myarr[np.isinf(myarr)] = 0.
myarr.sum()
# Exclue nan
np.nansum(myarr)
#%%

###############################################################################
# Chapter 4. Using NumPy C-API

###
# 5.2.4 weave
###
#%%
a = np.arange(1000)
b = np.arange(1000)
c = np.arange(1000)
d = 4*a + 5*a*b + 6*b*c
# d = np.empty(a.shape, "d")
# sp.weave.blitz("4*a + 5*a*b + 6*b*c")
#%%
