# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 19:43:22 2014

@author: phinary0
"""

# Fibonacci numbers module

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a+b
    print()

def fib2(n): # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result

# For command line execution

if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
