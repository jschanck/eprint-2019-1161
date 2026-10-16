# -*- coding: utf-8 -*-
"""
Useful functions that don't need to go into the appendix of the paper.
"""

from config import MagicConstants
from probabilities import C, Probabilities, _log_int
from mpmath import mp
import math
from utils import load_probabilities
import os
import csv


def fast_probabilities(d, n, k, beta=None, prec=None):
    """
    Useful probabilities.

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    # NOTE: This function is unused.

    prec = prec if prec else mp.prec

    with mp.workprec(prec):
        if beta is None:
            S1 = mp.exp(_log_int(d, n, k, None, 0.0, math.pi / 3))
            S2 = mp.exp(_log_int(d, n, k, None, math.pi / 3, math.pi))
            pf_ = S1 + S2
            ngr_ = C(d, mp.pi / 3)
            ngr_pf_ = S1
        elif beta < mp.pi / 6:
            pf_ = mp.exp(_log_int(d, n, k, float(beta), 0.0, math.pi))
            ngr_ = mp.mpf(1.0)
            ngr_pf_ = mp.mpf(1.0)
        else:
            beta = float(beta)
            S1 = mp.exp(_log_int(d, n, k, beta, 0.0, math.pi / 3))
            S2 = mp.exp(_log_int(d, n, k, beta, math.pi / 3, math.pi))
            S3 = mp.exp(_log_int(d, None, None, beta, 0.0, math.pi / 3))
            S4 = mp.exp(_log_int(d, None, None, beta, math.pi / 3, math.pi))
            ngr_ = S3 / (S3 + S4)
            pf_ = (S1 + S2) / (S3 + S4)
            ngr_pf_ = S1 / (S3 + S4)

        probs = Probabilities(
            d=d,
            n=n,
            k=k,
            ngr=ngr_,
            gr=1 - ngr_,
            pf=pf_,
            gr_pf=-1,
            ngr_pf=ngr_pf_,
            rho=-1,
            eta=1 - ngr_pf_ / ngr_,
            beta=beta,
            prec=prec,
        )
        return probs


def read_csv(filename, columns, read_range=None, ytransform=lambda y: y):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for i, row in enumerate(reader):
            if i == 0:
                columns = row.index(columns[0]), row.index(columns[1])
                continue
            data.append((int(row[columns[0]]), ytransform(float(row[columns[1]]))))

    if read_range is not None:
        data = [(x, y) for x, y in data if x in read_range]
    data = sorted(data)
    X = [x for x, y in data]
    Y = [y for x, y in data]
    return tuple(X), tuple(Y)


def linear_fit(filename, columns=("d", "log_cost"),
               low_index=0, high_index=100000, leading_coefficient=None):
    from scipy.optimize import curve_fit

    X, Y = read_csv(filename, columns=columns, read_range=range(low_index, high_index))

    if leading_coefficient is None:
        def f(x, a, b):
            return a * x + b
    else:
        def f(x, b):
            return leading_coefficient * x + b

    r = list(curve_fit(f, X, Y)[0])
    if leading_coefficient is not None:
        r = [leading_coefficient] + r
    print("{r[0]:.4}*x + {r[1]:.3}".format(r=r))
    return r
