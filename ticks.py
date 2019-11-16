"""
Module for formatting axis tick labels as multiples of PI (or other)
https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter


#class Multiple(object):
#    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
#        self.denominator = denominator
#        self.number = number
#        self.latex = latex
#​
#    def locator(self):
#        return plt.MultipleLocator(self.number / self.denominator)
#​
#    def formatter(self):
#        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def set_ticks(axis, major, minor):
    axis.set_major_locator(plt.MultipleLocator(major))
    axis.set_minor_locator(plt.MultipleLocator(minor))
    axis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
