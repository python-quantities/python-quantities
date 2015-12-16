import os
import sys

fn = sys.argv[1]
with open(fn) as f:
    s = f.read().lstrip('v').replace('-', '+', 1).replace('-', '.')
with open(fn, 'w') as f:
    f.write(s)
