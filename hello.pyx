# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:13:18 2014

@author: phinary0
"""

def say_hello_to(name):
    print("Hello %s!" % name)

cpdef double f(double x) except *:
    return x**2-x

def integrate_f(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b-a)/N
    for i in range(N):
        s += f(a+i*dx)
    return s * dx
