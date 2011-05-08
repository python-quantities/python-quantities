# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import, with_statement

import copy
import operator
import re
import threading


class _Config(object):

    @property
    def lock(self):
        return self._lock

    @property
    def use_unicode(self):
        with self.lock:
            return copy.copy(self._use_unicode)
    @use_unicode.setter
    def use_unicode(self, val):
        self._use_unicode = bool(val)

    def __init__(self):
        self._lock = threading.RLock()
        self._use_unicode = False

config = _Config()

superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

def superscript(val):
    # TODO: use a regexp:
    items = re.split(r'\*{2}([\d]+)(?!\.)', val)
    ret = []
    while items:
        try:
            s = items.pop(0)
            e = items.pop(0)
            ret.append(s+''.join(superscripts[int(i)] for i in e))
        except IndexError:
            ret.append(s)
    return ''.join(ret)

def format_units(udict):
    '''
    create a string representation of the units contained in a dimensionality
    '''
    num = []
    den = []
    keys = [k for k, o in
        sorted(
            [(k, k.format_order) for k in udict],
            key=operator.itemgetter(1)
        )
    ]
    for key in keys:
        d = udict[key]
        if config.use_unicode:
            u = key.u_symbol
        else:
            u = key.symbol
        if d>0:
            if d != 1:
                u = u + ('**%s'%d).rstrip('0').rstrip('.')
            num.append(u)
        elif d<0:
            d = -d
            if d != 1:
                u = u + ('**%s'%d).rstrip('0').rstrip('.')
            den.append(u)
    res = '*'.join(num)
    if len(den):
        if not res: res = '1'
        fmt = '(%s)' if len(den) > 1 else '%s'
        res = res + '/' + fmt%('*'.join(den))
    if not res: res = 'dimensionless'
    return res

def format_units_unicode(udict):
    res = format_units(udict)
    res = superscript(res)
    res = res.replace('**', '^').replace('*','·')

    return res
    
def format_units_latex(ustr,font='mathrm',mult=''):
    '''
    Replace the units string provided with an equivalent latex string.
    
    Division (a/b) will be replaced by \frac{a}{b}.
    
    Exponentiation (m**2) will be replaced with superscripts (m^{2})
    
    Multiplication (*) are replaced with the symbol specified by the mult argument.
    By default this is a blank string (no multiplication symbol).  Other useful
    options may be r'\cdot' or r'\*'
    
    The latex is set  with the font argument, and the default is the normal, 
    non-italicized font mathrm.  Other useful options include 'mathnormal', 
    'mathit', 'mathsf', and 'mathtt'.
    '''
    res = format_units(ustr)
    # Replace the first last parentheses with larger ones
    res = re.sub(r'^\(',r'\\left(',res)
    res = re.sub(r'\)$',r'\\right)',res)
    # Replace division (num/den) with \frac{num}{den}
    res = re.sub(r'(?P<num>\w+)/(?P<den>\w+)','\\\\frac{\g<num>}{\g<den>}',res)
    # Replace exponentiation (**exp) with ^{exp}
    res = re.sub(r'\*\*(?P<exp>\d+)',r'^{\g<exp>}',res)
    # Remove multiplication signs
    res = re.sub(r'\*',mult,res)
    return r'$\%s{%s}$' % (font,res)
    
