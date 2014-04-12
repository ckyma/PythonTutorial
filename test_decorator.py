# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:02:55 2014

@author: phinary0
"""

#%%
# the function takes a function as argument
def logger(func):
    def inner(*args, **kwargs): #1
        print("Arguments were :{0:s}, {1:s}".format(args, kwargs))
        return func(*args, **kwargs)    #2
    # the function returns a function
    return inner

@logger
def foo1(x, y = 1):
    return x * y
# equivalent to, i.e. wrap the function with a decorator
# foo1 = logger(foo1)

@logger
def foo2():
    return 2
# equivalent to, i.e. wrap the function with a decorator
# foo2 = logger(foo2)
#%%

foo1(5, 4)
foo1(5, y = 4)
foo1(1)
foo2()
