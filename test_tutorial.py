# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:28:14 2014

@author: phinary0
"""
###############################################################################
# Chapter 4. More Control Flow Tools

###
# 4.7.1. Default Argument Values
###

#%%
# The default values are evaluated at the point of function definition in the defining scope, 
# so that will print 5.
i = 5

def f(arg=i):
    print(arg)

i = 6
f()
#%%

#%%
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
#%%

###
# 4.7.2. Keyword Arguments
###

#%%
# formal parameter *name in tuple, final formal parameter **name in dictionary
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    keys = sorted(keywords.keys())
    for kw in keys:
        print(kw, ":", keywords[kw])
        
cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")
#%%

###
# 4.7.3. Arbitrary Argument Lists
###

#%%
# These arguments will be wrapped up in a tuple
def concat(*args, sep="/"):
    return sep.join(args)

concat("earth", "mars", "venus")
#%%

###
# 4.7.5. Lambda Expression
###
# Small anonymous functions can be created with the lambda keyword.
#%%
# 1. return a function
def make_incrementor(n):
    return lambda x: x + n
# x got initialized only once
f = make_incrementor(42)
# x = 42 always after
f(1)
#%%
#%%
# 2. pass a small function as an argument
pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
pairs
#%%

###
# 4.7.6. Documentation Strings
###
#%%
def my_function():
    """Do nothing, but document it.

    No, really, it doesn't do anything.
    """
    pass

print(my_function.__doc__)
#%%

###
# 4.7.7. Function Annotations
###
# Annotations are stored in the __annotations__ attribute of the function as a dictionary and have no effect on any other part of the function. 
# Parameter annotations are defined by a colon after the parameter name, followed by an expression evaluating to the value of the annotation. 
# Return annotations are defined by a literal ->, followed by an expression, between the parameter list and the colon denoting the end of the def statement. 
#%%
def f(ham: 42, eggs: int = 'spam') -> "Nothing to see here":
    print("Annotations:", f.__annotations__)
    print("Arguments:", ham, eggs)

f('wonderful')
#%%

###############################################################################
# Chapter 5. Data Structures

###
# 5.1.2. Using Lists as Queues
###
# Comparing to List, to implement a queue, use collections.deque which was designed to have fast appends and pops from both ends. 
#%%
from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")           # Terry arrives
queue.popleft()                 # The first to arrive now leaves
queue
#%%

###
# 5.1.3. List Comprehensions
###
# Common applications are to make new lists where each element is the result of some operations applied to each member of another sequence or iterable, 
# or to create a subsequence of those elements that satisfy a certain condition.
#%%
squares = []
for x in range(10):
    squares.append(x**2)
squares
#%%
# Alternative
#%%
squares = [x**2 for x in range(10)]
squares
#%%
#%%
squares = list(map(lambda x: x**2, range(10)))
squares
#%%

# A list comprehension consists of brackets containing an expression followed by a for clause, 
# then zero or more for or if clauses. The result will be a new list resulting from evaluating the expression in the context of the for and if clauses which follow it.
#%%
[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
#%%

###
# 5.3. Tuples and Sequences
###
# Tuple packing and sequence unpacking
#%%
t = 12345, 54321, 'hello!'
x, y, z = t
#%%

###
# 5.4. Sets
###
# A set is an unordered collection with no duplicate elements. 
# Basic uses include membership testing and eliminating duplicate entries. 
# Set objects also support mathematical operations like union, intersection, difference, and symmetric difference.
#%%
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)                      # show that duplicates have been removed
#%%
# Set operations
#%%
a = set('abracadabra')
b = set('alacazam')
a                                  # unique letters in a
a - b                              # letters in a but not in b
a | b                              # letters in either a or b
a & b                              # letters in both a and b
a ^ b                              # letters in a or b but not both
#%%

###############################################################################
# Chapter 6. Modules

#%%
# change the current working dir
# import os
# os.chdir("~/project-hg/python/phinarypython")
import fibo
fibo.fib2(100)
fibo.__name__
#%%

###
# 6.1. More on Modules
###
#%%
from fibo import fib
fib(500)
#%%
# Reload a module after changes
#%%
import imp; imp.reload(fibo)
#%%

###
# 6.1.3. “Compiled” Python files
###
# To speed up loading modules, not running of the program
#%%
# import compileall
# compileall.compile_dir('.', force=True)
# compileall.compile_file('fibo.py', force=True)
#%%

###
# 6.3. The dir() Function
###
# The built-in function dir() is used to find out which names a module defines.
#%%
dir(fibo)
#%%
# List currently defined all types of names: variables, modules, functions, etc.
#%%
dir()
#%%

###############################################################################
# Chapter 7. Input and Output

###
# 7.2.2. Saving structured data with json
###
#%%
import json
# console
json.dumps([1, 'simple', 'list'])
# file
fJson = open("test.json", "r+")
json.dump([1, 'simple', 'list'], fJson)
x = json.load(fJson)
x
fJson.close()
#%%

###############################################################################
# Chapter 8. Errors and Exceptions

###
# 8.4. Raising Exceptions
###
#%%
try:
    raise Exception('spam', 'eggs')
except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print(inst)          # __str__ allows args to be printed directly,
                         # but may be overridden in exception subclasses
    x, y = inst.args     # unpack args
    print('x =', x)
    print('y =', y)
#%%
#%%
raise NameError('HiThere')
#%%
#%%
try:
    raise NameError('HiThere')
except NameError:
    print('An exception flew by!')
    raise
#%%
    
###
# 8.5. User-defined Exceptions
###
#%%
class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

try:
    raise MyError(2*2)
except MyError as e:
    print('My exception occurred, value:', e.value)
    
raise MyError('oops!')
#%%

###
# 8.7. Predefined Clean-up Actions
###
#%%
with open("myfile.txt") as f:
    for line in f:
        print(line, end = "")
#%%
