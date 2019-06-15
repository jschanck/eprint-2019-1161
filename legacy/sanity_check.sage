#!/usr/bin/env sage
#! -*- coding: utf-8 -*-

import os
from optimise_popcnt import load_estimates


for filename in os.listdir('probabilities'):
    if filename == 'README.md':
        continue
    splitter = filename.find('_')
    d = int(filename[:splitter])
    n = int(filename[splitter+1:])
    all_estimates = load_estimates(d, n=n)
    for key, estimates in all_estimates.items():
        for probability, estimate in estimates.items():
            if estimate >= 1 or estimate <= 0:
                print filename, key, probabilitiy, estimate
