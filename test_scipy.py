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

###
# 1.5.2 Constrained minimization of multivariate scalar functions (minimize)
###
#%%
def func(x, sign=1.0):
    """ Objective function """
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)
def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])
cons = ({'type': 'eq',
         'fun' : lambda x: np.array([x[0]**3 - x[1]]),
        'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
        'jac' : lambda x: np.array([0.0, 1.0])})
res = sp.optimize.minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
                           constraints=cons, method='SLSQP', options={'disp': True})
print(res.x)
#%%

###
# 1.5.3 Least-square fitting (leastsq)
###
#%%
x = np.arange(0,6e-2,6e-2/30)
A,k,theta = 10, 1.0/3e-2, np.pi/6
y_true = A*np.sin(2*np.pi*k*x+theta)
y_meas = y_true + 2*np.random.randn(len(x))

def residuals(p, y, x):
    A,k,theta = p
    err = y-A*np.sin(2*np.pi*k*x+theta)
    return err
def peval(x, p):
    return p[0]*np.sin(2*np.pi*p[1]*x+p[2])
p0 = [8, 1/2.3e-2, np.pi/3]
print(np.array(p0))
plsq = sp.optimize.leastsq(residuals, p0, args=(y_meas, x))
plsq
print(plsq[0])
print(np.array([A, k, theta]))

import matplotlib.pyplot as plt
plt.plot(x,peval(x,plsq[0]),x,y_meas,'o',x,y_true)
plt.title('Least-squares fit to noisy data')
plt.legend(['Fit', 'Noisy', 'True'])
plt.show()
#%%

###
# 1.5.5 Root finding
###
#%%
from scipy.optimize import root
from numpy import cosh, zeros_like, mgrid, zeros

# parameters
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def residual(P):
   d2x = zeros_like(P)
   d2y = zeros_like(P)

   d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
   d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
   d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

   d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
   d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
   d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

   return d2x + d2y + 5*cosh(P).mean()**2

# solve
guess = zeros((nx, ny), float)
sol = root(residual, guess, method='krylov', options={'disp': True})
# sol = root(residual, guess, method='broyden2', options={'disp': True, 'max_rank': 50})
# sol = root(residual, guess, method='anderson', options={'disp': True, 'M': 10})
print('Residual: {0:g}'.format(abs(residual(sol.x)).max()))

# visualize
import matplotlib.pyplot as plt
x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
plt.pcolor(x, y, sol.x)
plt.colorbar()
plt.show()
#%%

## Precondition to accelerate
#%%
from scipy.optimize import root
from scipy.sparse import spdiags, kron
from scipy.sparse.linalg import spilu, LinearOperator
from numpy import cosh, zeros_like, mgrid, zeros, eye

# parameters
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def get_preconditioner():
    """Compute the preconditioner M"""
    diags_x = zeros((3, nx))
    diags_x[0,:] = 1/hx/hx
    diags_x[1,:] = -2/hx/hx
    diags_x[2,:] = 1/hx/hx
    Lx = spdiags(diags_x, [-1,0,1], nx, nx)

    diags_y = zeros((3, ny))
    diags_y[0,:] = 1/hy/hy
    diags_y[1,:] = -2/hy/hy
    diags_y[2,:] = 1/hy/hy
    Ly = spdiags(diags_y, [-1,0,1], ny, ny)

    J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly)

    # Now we have the matrix `J_1`. We need to find its inverse `M` --
    # however, since an approximate inverse is enough, we can use
    # the *incomplete LU* decomposition

    J1_ilu = spilu(J1)

    # This returns an object with a method .solve() that evaluates
    # the corresponding matrix-vector product. We need to wrap it into
    # a LinearOperator before it can be passed to the Krylov methods:

    M = LinearOperator(shape=(nx*ny, nx*ny), matvec=J1_ilu.solve)
    return M

def solve(preconditioning=True):
    """Compute the solution"""
    count = [0]

    def residual(P):
        count[0] += 1

        d2x = zeros_like(P)
        d2y = zeros_like(P)

        d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2])/hx/hx
        d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
        d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

        d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
        d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
        d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

        return d2x + d2y + 5*cosh(P).mean()**2

    # preconditioner
    if preconditioning:
        M = get_preconditioner()
    else:
        M = None

    # solve
    guess = zeros((nx, ny), float)

    sol = root(residual, guess, method='krylov',
               options={'disp': True,
                        'jac_options': {'inner_M': M}})
    print ('Residual', abs(residual(sol.x)).max())
    print ('Evaluations', count[0])

    return sol.x

def main():
    sol = solve(preconditioning=True)

    # visualize
    import matplotlib.pyplot as plt
    x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    plt.clf()
    plt.pcolor(x, y, sol)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
#%%
    
###############################################################################
# Chapter 1.9 Linear Algebra (scipy.linalg)

###
# 1.9.1 numpy.matrix vs 2D numpy.ndarray
##
# not recommended
#%%
A = np.mat([[1, 2], [3, 4]])
A.I
b = np.mat([5, 6])
b.T
A * b.T
#%%
# recommend
#%%
from scipy import linalg
A = np.array([[1,2],[3,4]])
linalg.inv(A)
b = np.array([[5], [6]])
b.T
# element product, not matrix product
A * b
# matrix multiplication
A.dot(b.T)
#%%

###
# 1.9.2 Basic routines
###
# Solving linear system
#%%
linalg.solve(A, b)
#%%
# Finding Determinant
#%%
linalg.det(A)
#%%

###
# 1.9.3 Decompositions
###

#Eigenvalues and eigenvectors
#%%
la, v = linalg.eig(A)
la
v
l1,l2 = la
v1 = np.array(v[:,0]).T
print(linalg.norm(A.dot(v1)-l1*v1))
#%%

###############################################################################
# 1.13 Statistics (scipy.stats)

from scipy import stats
from scipy.stats import norm

###
# 1.13.2 Random Variables
###
#%%
norm.cdf([-1., 0, 1])
stats.gamma(a=1, scale=2.).stats(moments="mv")
#%%

###############################################################################
# 1.16 Weave (scipy.weave)

# Only works in Python 2.7
from scipy import weave

