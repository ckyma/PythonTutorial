# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:28:14 2014

@author: phinary0
"""

###
# 4.7.1. Default Argument Values
###

# The default values are evaluated at the point of function definition in the defining scope, 
# so that will print 5.
i = 5
def f(arg=i):
    print(arg)
i = 6
f()

# Important warning: The default value is evaluated only once. 
# This makes a difference when the default is a mutable object such as a list, dictionary, or instances of most classes.
def f(a, L=[]):
    L.append(a)
    return L
print(f(1))
print(f(2))
print(f(3))

def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L
print(f(1))
print(f(2))
print(f(3))

