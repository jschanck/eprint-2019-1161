#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from collections import OrderedDict
from numpy import pi, sin
from scipy.integrate import quadrature as ty_gauss
from scipy.special import binom

import numpy as np
import random
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
    if abs(value) == 0:
        print 'it happened! it is a lucky day!'
        if value == -0.0:
            value = 0
    if value >= 0:
        return 1
    return 0


def uniform_iid_sphere(dim, num, popcnt_flag=False):
    """
    Samples points uniformly on the unit sphere in R^{dim}, i.e. S^{d - 1}.
    It samples from dim many zero centred Gaussians and normalises the vector.

    :param dim: the dimension of real space
    :param num: number of points to be sampled uniformly and i.i.d
    :param popcnt_flag: if ``False`` popcnt vectors come from the unit sphere,
                        else they mimic the behaviour of G6K
    :returns: an OrderedDict with keys which enumerate the points as values
    """
    sphere_points = OrderedDict()
    for point in range(num):
        # good randomness seems quite pertinent
        np.random.seed()

        if popcnt_flag:
            secure_random = random.SystemRandom()
            indices = secure_random.sample(range(dim), 6)
            plus = secure_random.sample(indices, 3)
            minus = [index for index in indices if index not in plus]
            vec = [0]*dim
            for index in plus:
                vec[index] = 1
            for index in minus:
                vec[index] = -1
            vec = np.array(vec)
        else:
            vec = np.random.randn(dim)
            vec /= np.linalg.norm(vec)

        sphere_points[point] = vec
    return sphere_points


def lossy_sketch(vec, popcnt):
    """
    Creates a lossy sketch of a point on the sphere a la the [Ducas18]
    extension to the [Charikar02] SimHash. The sketch will be called SimHash.

    In particular form a vector of the signs of the inner product of the point
    with some fixed points on the sphere.

    :param vec: the point on the sphere to make a SimHash of
    :param popcnt: the fixed vectors with which to make the SimHash
    :returns: an integer numpy array with values in {0, 1}, a SimHash
    """
    lossy_vec = np.zeros(len(popcnt))
    for index, popcnt_vec in popcnt.items():
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
    """
    Detailed stats output for all relevant probabilities given real dimension
    dim, number of vectors to trial num, number of popcnt vectors popcnt_num,
    popcnt threshold threshold and a switch whether the popcnt vectors should
    mimic G6K.
    """
    dim, num, popcnt_num, threshold, popcnt_flag = sys.argv[1:]

    dim = int(dim)
    num = int(num)
    popcnt_num = int(popcnt_num)
    threshold = int(threshold)
    popcnt_flag = eval(popcnt_flag)

    if popcnt_flag:
        print "Ignore integration based estimates!\n"

    if num <= 2:
        print "Please pick num > 2"
        return

    # initialise counters
    # (n)gr = (not) Gauss reduced, (n)pf = (did not) pass filter
    gr_pf = 0
    gr_npf = 0
    ngr_pf = 0
    ngr_npf = 0
    tot = 1./2. * num * (num - 1)

    # get dictionaries of points to be tested and used to form SimHashes
    popcnt = uniform_iid_sphere(dim, popcnt_num, popcnt_flag=popcnt_flag)
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

    est_gr = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3,
                      use_filt=False)
    est_pf = estimate(dim, popcnt_num, threshold)
    est_gr_pf = estimate(dim, popcnt_num, threshold, int_l=pi/3,
                         int_u=(2*pi)/3)
    est_gr_npf = estimate(dim, popcnt_num, threshold, int_l=pi/3,
                          int_u=(2*pi)/3, pass_filt=False)
    est_ngr_pf = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3)
    est_ngr_npf = 2*estimate(dim, popcnt_num, threshold, int_u=pi/3,
                             pass_filt=False)

    stats['gr'] = [gr, gr/float(tot), est_gr]
    stats['pf'] = [pf, pf/float(tot), est_pf]
    stats['gr_pf'] = [gr_pf, gr_pf/float(tot), est_gr_pf]
    stats['gr_npf'] = [gr_npf, gr_npf/float(tot), est_gr_npf]
    stats['ngr_pf'] = [ngr_pf, ngr_pf/float(tot), est_ngr_pf]
    stats['ngr_npf'] = [ngr_npf, ngr_npf/float(tot), est_ngr_npf]

    if gr == 0:
        print "None Gauss reduced, setting gr to 1, this should not happen"
        gr = 1
    elif gr == tot:
        print "All Gauss reduced, setting gr to tot - 1, pick a larger num"
        gr = tot - 1

    if pf == 0:
        print "Nothing passed filter, setting pf to 1, pick larger k"
        pf = 1
    elif pf == tot:
        print "Everything passed filter, setting pf to tot - 1, pick smaller k"
        pf = tot - 1

    # conditional probabilities, e.g. the first is gr given that pf
    stats['gr_pf/pf'] = [0, gr_pf/float(pf), est_gr_pf/est_pf]
    stats['gr_npf/npf'] = [0, gr_npf/float(tot - pf), est_gr_npf/(1 - est_pf)]
    stats['gr_pf/gr'] = [0, gr_pf/float(gr), est_gr_pf/est_gr]
    stats['ngr_pf/ngr'] = [0, ngr_pf/float(tot - gr), est_ngr_pf/(1 - est_gr)]

    mkeyl = 0
    for key in stats.keys():
        mkeyl = max(mkeyl, len(key))

    for key, value in stats.items():
        print key, " "*(mkeyl-len(key)), "\texp:", value[1], "\test:", value[2]
    print
    print '='*25


def gauss_test(dim, num):
    """
    Quick function to generate experimental results for the proportion of
    i.i.d. vectors on the unit sphere S^{dim - 1} \subset R^{dim} which are
    and are not Gauss reduced.

    :param dim: the dimension of real space
    :param num: the number of vectors to sample and test
    :returns: the proportion of vector pairs which are Gauss reduced
    """
    gr = 0
    tot = 1./2. * num * (num - 1)
    vector = uniform_iid_sphere(dim, num)
    for index1, vec1 in vector.items():
        for index2, vec2 in vector.items():
            if index1 <= index2:
                continue
            gr += 1*gauss_reduced(vec1, vec2)
    return float(gr)/tot


def filter_test(dim, num, popcnt_num, threshold, popcnt_flag=False):
    """
    Quick function to generate experimental results for the proportion of
    i.i.d. vectors on the unit sphere S^{dim - 1} \subset R^{dim} which do and
    do not pass the filter with popcnt_num tests and threshold k.

    :param dim: the dimension of real space
    :param num: the number of vectors to sample and test
    :param popcnt_num: the number of vectors with which to make the SimHash
    :param threshold: the acceptance/rejection threshold for popcnts
    :param popcnt_flag: if ``False`` popcnt vectors come from the unit sphere,
                        else they mimic the behaviour of G6K
    :returns: the proportion of vector pairs passing the filter
    """
    pf = 0
    tot = 1./2. * num * (num - 1)
    popcnt = uniform_iid_sphere(dim, popcnt_num, popcnt_flag=popcnt_flag)
    vector = uniform_iid_sphere(dim, num)
    for index, vec in vector.items():
        vector[index] = [vec, lossy_sketch(vec, popcnt)]
    for index1, vecs1 in vector.items():
        for index2, vecs2 in vector.items():
            if index1 >= index2:
                continue
            pf += hamming_compare(vecs1[1], vecs2[1], threshold)
    return float(pf)/tot


def estimate(dim, popcnt_num, threshold, int_l=0, int_u=pi, use_filt=True,
             pass_filt=True):
    """
    A function for computing the various probabilities we are interested in.

    If we are not considering the filter at all (e.g. when calculating Gauss
    reduced probabilities) set use_filt to ``False``, in which case pass_filt,
    popcnt_num and threshold are ignored.

    If we want to calculate probabilities when pairs do not pass the filter,
    set pass_filt to ``False``.

    :param dim: the dimension of real space
    :param popcnt_num: the number of vectors with which to make the SimHash
    :param threshold: the acceptance/rejection threshold for popcnts
    :param int_l: the lower bound of the integration in [0, pi]
    :param int_u: the upper bound of the integration in [0, pi]
    :param use_filt: boolean whether to consider the filter
    :param pass_filt: boolean whether to consider passing/failing the filter
    :returns: the chosen probability
    """
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
            if co == 0:
                continue

            def f(x): return (sin(x)**(d-2))*co*((x/pi)**i)*((1-(x/pi))**(n-i))
            prob += ty_gauss(f, int_l, int_u, tol=1.49e-8, rtol=1.49e-8,
                             maxiter=100)[0]
    else:

        def f(x): return (sin(x)**(d-2))
        prob = ty_gauss(f, int_l, int_u, tol=1.49e-8, rtol=1.49e-8,
                        maxiter=100)[0]

    def normaliser(dim):
        """
        The normalisation constant (dependent on dim) has a closed form!
        .. note:: we are interested in the relative surface area of
        (hyper)spheres on the surface of S^{dim - 1}, hence dim - 2.

        :param dim: the dimension of real space
        :returns: the normalisation constant for the integral estimates
        """
        d = dim - 2
        norm = 1
        if d % 2 == 0:
            for i in range(1, d + 1):
                if i % 2 == 0:
                    norm *= float(i**(-1))
                else:
                    norm *= i
            norm *= pi
        else:
            for i in range(2, d + 1):
                if i % 2 == 0:
                    norm *= i
                else:
                    norm *= float(i**(-1))
            norm *= 2
        return 1./norm

    return normaliser(d) * prob


def filter_wrapper(dims, num, popcnt_nums, thresholds):
    """
    Quick check of accuracy of filter proportion estimates for lists of dims,
    popcnt_nums and thresholds to try. Prints basic stats.

    :param dims: a list of dim to be tried
    :param num: the number of vectors to sample and test
    :param popcnt_nums: a list of popcnt_num to be tried
    :param thresholds: a list of thresholds to be tried
    """
    if type(dims) is not list:
        dims = [dims]
    if type(popcnt_nums) is not list:
        popcnt_nums = [popcnt_nums]
    if type(thresholds) is not list:
        thresholds = [thresholds]
    for dim in dims:
        for popcnt_num in popcnt_nums:
            for threshold in thresholds:
                print
                print "dim %d, popcnt_num %d, threshold %d" % (dim,
                                                               popcnt_num,
                                                               threshold)
                pf = filter_test(dim, num, popcnt_num, threshold)
                est = estimate(dim, popcnt_num, threshold)
                print "(exp, est): (%.6f, %.6f)" % (pf, est)
                print "="*25


def gauss_wrapper(dims, num):
    """
    Quick check of accuracy of reduced proportion estimates for lists of dims.
    Prints basic stats.

    :param dims: a list of dim to be tried
    :param num: the number of vectors to sample and test
    """
    if type(dims) is not list:
        dims = [dims]
    for dim in dims:
        print
        print "dim %d" % dim
        gr = gauss_test(dim, num)
        est = estimate(dim, 0, 0, int_l=pi/3, int_u=(2*pi)/3, use_filt=False)
        print "(exp, est): (%.6f, %.6f)" % (gr, est)
        print "="*25


if __name__ == '__main__':
    main()
