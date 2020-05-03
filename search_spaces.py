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

    def cpolyt(theta):
        # Eq 12.3 Thijs' thesis: bucketing probability in terms of d, theta
        bucket_prob = mp.exp(-mp.log(d) * mp.tan(theta / mp.mpf(2))**mp.mpf(2))
        return bucket_prob * A(d, theta, prec=mp.prec)

    def cpolyt_int(theta):
        return mp.quad(cpolyt, (mp.mpf(0), theta), error=True)[0]

    # prob for fixed u a uniform v on sphere will fall into the same bucket
    expected = cpolyt_int(mp.pi)
    # prob for fixed u a uniform v falls into k buckets
    expected_k = expected**k
    # prob for fixed u a uniform v falls into at least one of t `k-AND buckets'
    expected_k_t = mp.mpf(1) - (mp.mpf(1) - expected_k)**t

    def dtheta_expected_k_t(theta):
        # differentiate 1 - (1 - (cpolyt_int(theta))**k)**t wrt theta
        return k * t * (1 - cpolyt_int(theta)**k)**(t-1) * cpolyt_int(theta)**(k-1) * cpolyt(theta)

    # eta = 1 - P[P_{k, n} | ngr AND falls into the same cpolyt bucket]
    eta_n = mp.quad(lambda x: dtheta_expected_k_t(x) * P(n_pop, k_pop, x),
                    (0, mp.pi / 3), error=True)[0]
    eta_d = mp.quad(dtheta_expected_k_t, (0, mp.pi / 3), error=True)[0]

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
