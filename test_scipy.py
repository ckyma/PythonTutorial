# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 18:14:01 2014

@author: Yu
"""

###############################################################################
# Chapter 1.2 Basic functions

import numpy as np
import scipy as sp
import matplotlib as mpl

###
# 1.2.1 Interaction with Numpy
###

## Index Tricks
#%%
np.concatenate(([3], [0]*5, np.arange(-1, 1.002, 2/9.0)))
# shortcut row concatenation
np.r_[3, [0]*5, -1:1:10j]
# shortcut column concatenation
np.c_[[3]*5, [0]*5, -1:1:5j]

# mgrid
# 1. same as arange
np.arange(5)
np.mgrid[0:5]
# 2. mesh grid array
np.mgrid[0:5:4j, 0:5:4j]
# a more efficient way for mesh grid only
np.ogrid[0:5:4j, 0:5:4j]
#%%

## Polynomials
#%%
p = np.poly1d([3, 4, 5])
p
print(p)
print(p*p)
# Integral, with constant = 6
print(p.integ(k = 6))
# Defferentiation
print(p.deriv())
# Evaluste at x = 4 and 5 respectly
p([4, 5])
#%%

## Vectorizing functions
#%%
def addsubtract(a, b):
    if a > b:
        return a - b
    else:
        return a + b

vec_addsubtract = np.vectorize(addsubtract)
vec_addsubtract([0, 3, 6, 9], [1, 3, 5, 7])
#%%

## Type handling
#%%
np.isscalar(1)
np.isscalar(np.arange(2))
# type casting
d = np.arange(5)
np.cast["f"](d)
#%%

## Other useful functions
#%%
# Select
x = np.r_[-2:3]
x
print([0, x+2])
np.select([x > 3, x >= 0], [0, x+2])
np.select([x > 3], [x+2])
np.select([x >= 0], [x+2])
# scipy.misc
import scipy.misc
# from scipy import misc
sp.misc.factorial(5)
sp.misc.comb(100, 5)
#%%

###############################################################################
# Chapter 1.3 Special functions

###
# Bessel functions of real order(jn, jn_zeros)
###
#%%
from scipy.special import jn, jn_zeros
def drumhead_height(n, k, distance, angle, t):
    nth_zero = jn_zeros(n, k)
    return np.cos(t) * np.cos(n * angle) * jn(n, distance * nth_zero)
theta = np.r_[0:2*np.pi:50j]
radius = np.r_[0:1:50j]
x = np.array([r*np.cos(theta) for r in radius])
y = np.array([r*np.sin(theta) for r in radius])
z = np.array([drumhead_height(1, 1, r, theta, 0.5) for r in radius])

import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = pylab.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = cm.jet)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
pylab.show()
pylab.close()
#%%

###############################################################################
# Chapter 1.4 Integration (scipy.integrate)

import scipy.integrate
help(sp.integrate)

###
# 1.4.1 General integration (quad)
###
#%%
result = sp.integrate.quad(lambda x: sp.special.jv(2.5, x), 0, 4.5)
print(result)
#%%
# User defined function and range
#%%
def integrand(x, a, b):
    return a * x + b
a = 2
b = 1
I = sp.integrate.quad(integrand, 0, 1, args = (a, b))
I

# Vectorize
def integrand(t, n, x):
    return np.exp(-x * t) / t**n
    
def expint(n ,x):
    return sp.integrate.quad(integrand, 1, Inf, args=(n, x))[0]
    
vec_expint = np.vectorize(expint)
vec_expint(3, np.arange(1.0, 4.0, 0.5))

sp.special.expn(3, np.arange(1.0, 4.0, 0.5))
#%%

###############################################################################
# Chapter 1.5 Optimization (scipy.optimize)

import scipy.optimize
help(scipy.optimize)

###
# 1.5.1 Unconstrained minimization of multivariate scalar functions (minimize)
###
## Nelder-Mead Simplex algorithm (method='Nelder-Mead')
#%%
def rosen(x):
    """The Rosen function"""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = sp.optimize.minimize(rosen, x0, method = "nelder-mead", 
                           options = {"xtol": 1e-8, "disp": True})
print(res.x)
#%%
## Powell's method
#%%
resP = sp.optimize.minimize(rosen, x0, method = "powell", 
                           options = {"xtol": 1e-8, "disp": True})
print(resP.x)
#%%
