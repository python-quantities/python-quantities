# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import, with_statement

import copy
import operator
import re
import threading
try:
    import user
except ImportError:
    user = None


class _Config(object):

    @property
    def lock(self):
        return self._lock

    def _get_use_unicode(self):
        with self.lock:
            return copy.copy(self._use_unicode)
    def _set_use_unicode(self, val):
        self._use_unicode = bool(val)
    use_unicode = property(_get_use_unicode, _set_use_unicode)

    def __init__(self):
        self._lock = threading.RLock()
        self._use_unicode = getattr(user, 'quantities_unicode', False)

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
