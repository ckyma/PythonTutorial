# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:28:14 2014

@author: ckyma
"""

#%%
###############################################################################
# Chapter 1. Whetting Your Appetite

# 1. Python allows you to split your program into modules that can be reused in other Python programs.
# 2. You can link the Python interpreter into an application written in C and use it as an extension or command language for that application.

#%%
###############################################################################
# Chapter 2. Using the Python Interpreter

# the script name and additional arguments thereafter are turned into a list of strings and assigned to the argv variable in the sys module.
# You can access this list by executing import sys. 
# The length of the list is at least one; when no script and no arguments are given, sys.argv[0] is an empty string. 
# When the script name is given as '-' (meaning standard input), sys.argv[0] is set to '-'. 
# When -c command is used, sys.argv[0] is set to '-c'. 
# When -m module is used, sys.argv[0] is set to the full name of the located module. 
# Options found after -c command or -m module are not consumed by the Python interpreter’s option processing but left in sys.argv for the command or module to handle.

#%%
###############################################################################
# Chapter 3. An Informal Introduction to Python

# Comments in Python start with the hash character, #, and extend to the end of the physical line. 
# A comment may appear at the start of a line or following whitespace or code, but not within a string literal. 
# A hash character within a string literal is just a hash character.

###
# 3.1 Using Python as a Calculator
###

###
# 3.1.1 Numbers
###

#%%

16 / 2 # Division always retuens a float, i.e., 8.0

16 // 3 # get floor, i.e., 5
16 % 3 # get remainder, i.e., 1

2 ** 7 # 2 to the power of 7, i.e., 128

# In interactive mode ONLY, the last printed expression is assigned to the variable _
tax = 12.5 / 100
price = 100.5
price * tax
# price + _ # 113.0625
# round(_, 2) # 113.06

complexNum = 3+5j # complex type number

#%%

###
# 3.1.2 Strings
###
#%%

s = '"First" line.\nSecond line.'
s # Still \n in output
print(s) # Formatted output

print(r'c:\name')



#%%

###
# 3.1.3 Lists
###
#%%

# List supports concatenation
[1, 2] + [3, 4]

# Nested / Multidimensional list
multiList = [[11, 12], [21, 22]]
multiList[0][0]


#%%

#%%
###############################################################################
# Chapter 4. More Control Flow Tools

###
# 4.1 if Statements
###

x = int(input("Please enter an interger: "))

if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
else:
    print('More')

###
# 4.2 for Statements
###

words = ['ab', 'cd', 'efg']

# Make a copy of the list using ':' when loop and modify the list
for w in words[:]:
    if len(w) > 2:
        words.insert(0, w)

print(words)

###
# 4.3 The range() Function
###

for i in range(len(words)):
    print(i, words[i])

print(list(range(10)))
    
###
# 4.6 Defining Functions
###

def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a + b
    print()

fib(2000)
fib

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
with open("test.json") as f:
    for line in f:
        print(line, end = "")

#%%

###############################################################################
# Chapter 9. Classes

###
# 9.6. Private Variables
###
#%%
class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update   # private copy of original update() method

class MappingSubclass(Mapping):

    def update(self, keys, values):
        # provides new signature for update()
        # but does not break __init__()
        for item in zip(keys, values):
            self.items_list.append(item)
#%%

###
# 9.8. Exceptions Are Classes Too
###
#%%
class B(Exception):
    pass
class C(B):
    pass
class D(C):
    pass

for cls in [B, C, D]:
    try:
        raise cls()
    except D:
        print("D")
    except C:
        print("C")
    except B:
        print("B")

for cls in [B, C, D]:
    try:
        raise cls()
    except B:
        print("B")
    except C:
        print("C")
    except D:
        print("D")
#%%

###
# 9.9. Iterators
###
#%%
for element in [1, 2, 3]:
    print(element)
for element in (1, 2, 3):
    print(element)
for key in {'one':1, 'two':2}:
    print(key)
for char in "123":
    print(char)
for line in open("test.json"):
    print(line)
#%%
#%%
class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

rev = Reverse('spam')
iter(rev)
for char in rev:
    print(char)
#%%

###
# 9.9. Generators
###
#%%
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
        
for char in reverse('golf'):
    print(char)
#%%
    
###
# 9.10. Generator Expressions
###
#%%
sum(i*i for i in range(10))
#%%

###############################################################################
# Chapter 10. Brief Tour of the Standard Library

###
# 10.8. Dates and Times
###
#%%
from datetime import date
now = date.today()
now
#%%
#%%
birthday = date(1983, 7, 27)
age = now - birthday
age.days
#%%

###
# 10.9. Data Compression
###
#%%
from timeit import Timer
Timer("t = a; a = b; b = t", "a = 1; b = 2").timeit()
Timer("a, b = b, a", "a = 1; b = 2").timeit()
#%%

###
# 10.11. Quality Control
###
# doctest in docstring
#%%
def average(values):
    """Computes the arithmatic mean of a list of numbers.
    
    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)
    
import doctest
doctest.testmod()
#%%
# unittest
#%%
import unittest

class TestStaticalFunctions(unittest.TestCase):
    
    def test_average(self):
        self.assertEqual(average([20, 30, 70]), 40.0)
        self.assertEqual(round(average([1, 5, 7]), 1), 4.3)
        with self.assertRaises(ZeroDivisionError):
            average([])
        with self.assertRaises(TypeError):
            average(20, 30, 70)
            
unittest.main() # Calling from the command line invokes all tests

#%%

###############################################################################
# Chapter 11. Brief Tour of the Standard Library - Part II

###
# 11.4. Multi-threading
###
#%%
import threading, zipfile
class AsyncZip(threading.Thread):
    def __init__(self, infile, outfile):
        threading.Thread.__init__(self)
        self.infile = infile
        self.outfile = outfile
    def run(self):
        f = zipfile.ZipFile(self.outfile, "w", zipfile.ZIP_DEFLATED)
        f.write(self.infile)
        f.close()
        print("Finished background zip of:", self.infile)
        
background = AsyncZip("test.json", "test.zip")
background.start()
print("The main process continues to run in foreground.")

background.join()   # Wait for the background task to finish
print("Main program waited until background was done.")
#%%

###
# 11.5. Logging
###
#%%
import logging
logging.debug("Debugging information")
logging.warning("Warning: config file %s not found", "server.conf")
logging.error("Error occurred")
#%%

###
# 11.6. Weak References
###
#%%
import weakref, gc
class A:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)
        
a = A(10)
d = weakref.WeakValueDictionary()
d["primary"] = a
d["primary"]
del a
d["primary"]
gc.collect()
d["primary"]
#%%

###
# 11.7. Tools for Working with Lists
###
#%%
from heapq import heapify, heappop, heappush
data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
heapify(data)                       # rearrange the list into heap order
data
heappush(data, -5)                  # add a new entry
data
[heappop(data) for i in range(3)]   # fetch the three smallest entries
data
#%%

###
# 11.7. Tools for Working with Lists
###
#%%
import decimal
round(decimal.Decimal("0.70") * decimal.Decimal("1.05"), 2)
round(.70 * 1.05, 2)
#%%
#%%
sum([decimal.Decimal("0.1")] * 10) == decimal.Decimal("1.0")
sum([.1] * 10) == 1.0
#%%
#%%
decimal.getcontext().prec = 36
decimal.Decimal("1") / decimal.Decimal("7")
#%%
