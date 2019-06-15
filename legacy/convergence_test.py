#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpmath import mp
from optimise_popcnt import maximise_optimising_func

import os


popcnts = []
dim = 80
for filename in os.listdir("probabilities"):
    if filename[:3] == '80_':
        popcnts += [int(filename[3:])]

for popcnt in popcnts:
    print maximise_optimising_func(dim, popcnt_num=popcnt)

print (mp.ceil(2**(.2075 * dim)))**.5
