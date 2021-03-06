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
# Only works under python 2.7
#%%
a = np.arange(1000)
b = np.arange(1000)
c = np.arange(1000)
# d = 4*a + 5*a*b + 6*b*c
d = np.empty(a.shape, "d")
sp.weave.blitz("d = 4*a + 5*a*b + 6*b*c")
#%%
# inline C-code
#%%
code = r"""
int i;
py::tuple results(2);
for (i=0; i<a.length(); i++) {
    a[i] = i;
}
results[0] = 3.0;
results[1] = 4.0;
return_val = results;
"""
a = [None]*10
res = weave.inline(code, ["a"])
#%%
