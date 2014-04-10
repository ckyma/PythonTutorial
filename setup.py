# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:14:00 2014

@author: phinary0
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Hello world app',
  ext_modules = cythonize("hello.pyx"),
)

# Next, build in command line
# python setup.py build_ext --inplace