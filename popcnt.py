#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from collections import OrderedDict
from math import pi, sin
from scipy.special import binom

import numpy as np
import scipy.integrate as scint
import sys


def gauss_reduced(vec1, vec2):
    """
    Check whether two vectors sampled i.i.d. uniformly from the unit sphere are
    Gauss reduced.
    .. note:: This test is only true for vectors of equal norm!

    :param vec1: the first vector on the sphere
    :param vec2: the second vector on the sphere
    :returns: ``True`` if Gauss reduced and ``False`` otherwise
    """
    cos_theta = np.inner(vec1, vec2)
    if 0.5 < abs(cos_theta) < 1:
        return False
    return True


def sign(value):
    """
    Utility function for computing the sign of real number.

    :param value: a float or int
    :returns: ``True`` if value is >= 0, else ``False``
    """
    if value >= 0:
        return 1
    return 0


def uniform_iid_sphere(dim, num):
    """
    Samples points uniformly on the unit sphere in R^{dim}, i.e. S^{d - 1}.
    It samples from dim many zero centred Gaussians and normalises the vector.

    :param dim: the dimension of the unit sphere
    :param num: number of points to be sampled uniformly and i.i.d
    :returns: an OrderedDict with keys which enumerate the points as values
    """
    sphere_points = OrderedDict()
    for point in range(num):
        np.random.seed()
        vec = np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        sphere_points[point] = vec
    return sphere_points


def lossy_sketch(vec, popcnt_dict):
    """
    Creates a lossy sketch of a point on the sphere a la the [Ducas18]
    extension to the [Charikar02] SimHash. The sketch will be called SimHash.

    In particular form a vector of the signs of the inner product of the point
    with some fixed points on the sphere.

    :param vec: the point on the sphere to make a SimHash of
    :param popcnt_dict: the fixed vectors with which to make the SimHash
    :returns: an integer numpy array with values in {0, 1}, a SimHash
    """
    lossy_vec = np.zeros(len(popcnt_dict))
    for index, popcnt_vec in popcnt_dict.items():
        lossy_vec[index] = int(sign(np.inner(vec, popcnt_vec)))
    return lossy_vec


def hamming_compare(lossy_vec1, lossy_vec2, threshold):
    """
    Takes an XOR of two SimHashes and checks the popcnt is above= or below= a
    threshold.

    :param lossy_vec1: the SimHash of the first point on the sphere
    :param lossy_vec2: the SimHash of the second point on the sphere
    :param threshold: the acceptance/rejection threshold for popcnts
    :returns: ``True`` if the pair passed the filter, ``False`` otherwise
    """
    assert len(lossy_vec1) == len(lossy_vec2), "Different dimension sketches!"
    popcnt_num = len(lossy_vec1)
    hamming_xor = sum(np.bitwise_xor(lossy_vec1.astype(int),
                                     lossy_vec2.astype(int)))
    if hamming_xor <= threshold or hamming_xor >= popcnt_num - threshold:
        return True
    return False


def main():
    dim, num, popcnt_num, threshold = sys.argv[1:]

    dim = int(dim)
    num = int(num)
    popcnt_num = int(popcnt_num)
    threshold = int(threshold)

    # initialise counters
    # (n)gr = (not) Gauss reduced, (n)pf = (did not) pass filter
    gr_pf = 0
    gr_npf = 0
    ngr_pf = 0
    ngr_npf = 0
    tot = 1./2. * num * (num - 1)

    # get dictionaries of points to be tested and used to form SimHashes
    popcnt = uniform_iid_sphere(dim, popcnt_num)
    vector = uniform_iid_sphere(dim, num)

    # create SimHashes
    for index, vec in vector.items():
        vector[index] = [vec, lossy_sketch(vec, popcnt)]

    # compare pairwise different vectors and their SimHashes
    for index1, vecs1 in vector.items():
        for index2, vecs2 in vector.items():
            if index1 >= index2:
                continue
            gauss_check = gauss_reduced(vecs1[0], vecs2[0])
            lossy_check = hamming_compare(vecs1[1], vecs2[1], threshold)

            # information about filter pass/fail and Gauss reduction for pair
            if gauss_check and lossy_check:
                gr_pf += 1
            if gauss_check and not lossy_check:
                gr_npf += 1
            if not gauss_check and lossy_check:
                ngr_pf += 1
            if not gauss_check and not lossy_check:
                ngr_npf += 1

    gr = gr_pf + gr_npf
    pf = gr_pf + ngr_pf

    # collect all stats, of form [absolute value, tot_ratio, est]
    stats = OrderedDict()
    est = 1-2*estimate(dim, popcnt_num, threshold, int_u=pi/3, use_filt=False)
    stats['gr'] = [gr, gr/float(tot), est]
    est = estimate(dim, popcnt_num, threshold)
    stats['pf'] = [pf, pf/float(tot), est]
    est = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3)
    stats['gr_pf'] = [gr_pf, gr_pf/float(tot), est]
    est = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3,
                   pass_filt=False)
    stats['gr_npf'] = [gr_npf, gr_npf/float(tot), est]
    est = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3)
    stats['ngr_pf'] = [ngr_pf, ngr_pf/float(tot), est]
    est = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3, pass_filt=False)
    stats['ngr_npf'] = [ngr_npf, ngr_npf/float(tot), est]

    est1 = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3)
    est2 = estimate(dim, popcnt_num, threshold)
    est = est1/float(est2)
    stats['gr_pf/pf'] = [None, gr_pf/float(pf), est]
    est1 = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3,
                    pass_filt=False)
    est2 = 1 - estimate(dim, popcnt_num, threshold)
    est = est1/float(est2)
    stats['gr_npf/npf'] = [None, gr_npf/float(tot - pf), est]

    est1 = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3)
    est2 = 1-2*estimate(dim, popcnt_num, threshold, int_u=pi/3, use_filt=False)
    est = est1/float(est2)
    stats['gr_pf/gr'] = [None, gr_pf/float(gr), est]
    est1 = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3)
    est2 = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3, use_filt=False)
    est = est1/float(est2)
    stats['ngr_pf/ngr'] = [None, ngr_pf/float(tot - gr), est]

    for key, value in stats.items():
        print key, "\texp_ratio\t", value[1], "\test_ratio\t", value[2]
    print
    print '-------------------------------------------------------------------'


def gauss_test(dim, num):
    """
    Quick function to generate experimental results for the proportion of
    i.i.d. vectors on the unit sphere S^{dim - 1} \subset R^{dim} which are
    and are not Gauss reduced.

    :param dim: the dimension of real space
    :param num: the number of vectors to sample and test
    """
    counter = 0
    compar_tot = float(num) * float(num - 1) * 0.5
    gauss_dict = uniform_iid_sphere(dim, num)
    for index1, vec1 in gauss_dict.items():
        for index2, vec2 in gauss_dict.items():
            if index1 <= index2:
                continue
            counter += 1*gauss_reduced(vec1, vec2)
    print 'Fraction already reduced:', float(counter)/float(compar_tot)
    print 'Fraction not yet reduced:', 1 - float(counter)/float(compar_tot)


def filter_test(dim, num, popcnt_num, threshold):
    popcnt_pass = 0
    popcnt_tot = 1./2. * num * (num - 1)
    popcnt_dict = uniform_iid_sphere(dim, popcnt_num)
    vec_dict = uniform_iid_sphere(dim, num)
    for index, vec in vec_dict.items():
        vec_dict[index] = [vec, lossy_sketch(vec, popcnt_dict)]
    for index1, vecs1 in vec_dict.items():
        for index2, vecs2 in vec_dict.items():
            if index1 >= index2:
                continue
            passed = hamming_compare(vecs1[1], vecs2[1], threshold)
            popcnt_pass += passed
    return float(popcnt_pass)/popcnt_tot


def estimate(dim, popcnt_num, threshold, int_l=0, int_u=pi, use_filt=True,
             pass_filt=True):
    d = dim
    n = popcnt_num
    k = threshold

    if use_filt:
        if pass_filt:
            coeffs = [binom(n, i) for i in range(0, k + 1)]
            coeffs += [0] * (n - (2 * k) - 1)
            coeffs += [binom(n, i) for i in range(n - k, n + 1)]
        else:
            coeffs = [0]*(k + 1)
            coeffs += [binom(n, i) for i in range(k + 1, n - k)]
            coeffs += [0]*(k + 1)

        prob = 0
        for i in range(n + 1):
            co = coeffs[i]

            def f(x): return (sin(x)**(d-1))*co*((x/pi)**i)*((1-(x/pi))**(n-i))
            prob += scint.quad(f, int_l, int_u)[0]
    else:

        def f(x): return (sin(x)**(d-1))
        prob = scint.quad(f, int_l, int_u)[0]

    def n(x): return (sin(x)**(d-1))

    norm = 1./scint.quad(n, 0, pi)[0]
    return norm * prob


def filter_wrapper(dims, num, popcnt_nums, thresholds):
    for dim in dims:
        for popcnt_num in popcnt_nums:
            for threshold in thresholds:
                print "dim %d, popcnt_num %d, threshold %d" % (dim,
                                                               popcnt_num,
                                                               threshold)
                filter_pass = filter_test(dim, num, popcnt_num, threshold)
                est = estimate(dim, popcnt_num, threshold)
                print "experimental %.6f, estimate %.6f" % (filter_pass, est)
                print
                print "======================================================="


if __name__ == '__main__':
    main()
