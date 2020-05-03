# -*- coding: utf-8 -*-

import csv
import os

from mpmath import mp

from probabilities import A, C, P, W

mp.prec = 212


def cpolyt_search_size(d, metric='dw'):
    d = mp.mpf(d)
    # c_t for quantum cross-poly found in §14.2.9 Thijs' thesis
    c_t = mp.mpf(.0595)
    # relationship between c_t and k found in §12.3.1 Thijs' thesis
    k = mp.floor(mp.mpf(3) * c_t * d / mp.log(d, 2))
    # log_2 number of buckets = c_t * d + o(d)
    t = mp.floor(mp.mpf(2)**(c_t * d))

    # use same popcount params as equivalent BDGL instance
    filename = os.path.join("..", "data", "cost-estimate-{f}-{metric}.csv")
    filename = filename.format(f="list_decoding", metric=metric)
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"',
                               quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            if row[0] == str(int(d)):
                n_pop = int(row[1])
                k_pop = int(row[2])

    def int_f(theta):
        # Eq 12.3 Thijs' thesis
        cpolyt_prob = mp.exp(-mp.log(d) * mp.tan(theta / mp.mpf(2))**mp.mpf(2))
        return cpolyt_prob * A(d, theta, prec=mp.prec)

    def int_pop_f(theta):
        # Eq 12.3 Thijs' thesis
        cpolyt_prob = mp.exp(-mp.log(d) * mp.tan(theta / mp.mpf(2))**mp.mpf(2))
        return P(n_pop, k_pop, theta) * cpolyt_prob * A(d, theta, prec=mp.prec)

    # prob for fixed u a uniform v on sphere will fall into the same bucket
    expected = mp.quad(int_f, (mp.mpf(0), mp.pi), error=True)[0]
    # prob for fixed u a uniform v falls into all k buckets
    expected_k = expected**k
    # prob for fixed u a uniform v falls into at least one of t `k-AND buckets'
    expected_k_t = mp.mpf(1) - (mp.mpf(1) - expected_k)**t

    # eta = 1 - P[P_{k, n} | ngr AND falls into the same cpolyt bucket]
    ngr_l = mp.mpf(0)
    ngr_u = mp.pi / mp.mpf(3)

    eta_n = mp.quad(int_pop_f, (ngr_l, ngr_u), error=True)[0]
    eta_d = mp.quad(int_f, (ngr_l, ngr_u), error=True)[0]

    eta = mp.mpf(1) - eta_n / eta_d

    # not multiplying by N, just 1 / (1 - eta)
    return (mp.mpf(1) / (mp.mpf(1) - eta)) * expected_k_t


def bdgl_search_size(d, metric="dw"):
    filename = os.path.join("..", "data", "cost-estimate-{f}-{metric}.csv")
    filename = filename.format(f="list_decoding", metric=metric)
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar='"',
                               quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            if row[0] == str(d):
                theta = mp.mpf(float(row[3]))
                eta = mp.mpf(row[7])

    C_ = C(d, theta, prec=mp.prec)
    t_ = mp.mpf(1) / W(d, theta, theta, mp.pi / mp.mpf(3))
    # just multiplying by 1 / (1 - eta)
    return (mp.mpf(1.) / (mp.mpf(1.) - eta)) * t_ * C_**mp.mpf(2)


for i in range(64, 1024+1, 16):
    print(i, float(mp.log(cpolyt_search_size(i), 2)),
          float(mp.log(bdgl_search_size(i), 2)))
