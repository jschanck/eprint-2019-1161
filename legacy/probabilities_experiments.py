# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from collections import OrderedDict
from probabilities import pf as pff
from probabilities import ngr as ngrf


def gauss_reduced(u, v):
    """
    Check whether two vectors sampled i.i.d. uniformly from the unit sphere are
    Gauss reduced.

    .. note:: This test is only true for vectors of equal norm!

    :param u: the first vector on the sphere
    :param v: the second vector on the sphere
    :returns: ``True`` if Gauss reduced and ``False`` otherwise
    """
    cos_theta = np.inner(u, v)
    if 0.5 < abs(cos_theta) < 1:
        return False
    return True


def sgn(value):
    """
    Utility function for computing the sign of real number.

    :param value: a float or int
    :returns: ``True`` if value is >= 0, else ``False``
    """
    if abs(value) == 0:
        print('it happened! it is a lucky day!')
        if value == -0.0:
            value = 0
    if value >= 0:
        return 1
    return 0


def uniform_iid_sphere(dp1, num, spherical_code=False):
    """
    Samples ``num`` points uniformly on the unit sphere in `R^{d+1}`, i.e. `S^{d}`.
    It samples from `(d+1)` many zero centred Gaussians and normalises the vector.

    :param dp1: the dimension of real space, `d+1`
    :param num: number of points to be sampled uniformly and i.i.d
    :param spherical_code: if ``False`` popcnt vectors come from the unit sphere,
                        else they mimic the behaviour of G6K
    :returns: an OrderedDict with keys which enumerate the points as values
    """
    sphere_points = OrderedDict()

    if not spherical_code:
        for point in range(num):
            # good randomness seems quite pertinent
            vec = np.random.randn(dp1)
            vec /= np.linalg.norm(vec)
            sphere_points[point] = vec
    else:
        with open("spherical_coding/sc_{dp1}_{num}.def".format(dp1=dp1, num=num)) as f:
            for point, line in enumerate(f.readlines()): # noqa
                idx = map(int, line.split(" "))
                vec = [0]*dp1
                for i in idx[:3]:
                    vec[i] = 1
                for i in idx[3:]:
                    vec[i] = -1
                sphere_points[point] = np.array(vec)
    return sphere_points


def biased_sphere(dp1, num, num_filters, delta):
    """

    :param dp1: the dimension of real space, `d+1`
    :param num: number of points to be sampled uniformly and i.i.d
    :param delta:
    :param num_filters: number of filters to use
    :returns: an OrderedDict with keys which enumerate the points as values

    """
    sphere_points = OrderedDict()
    point = 0

    w = []
    j = 0
    for i in range(num_filters):
        w_ = np.random.randn(dp1)
        w_ /= np.linalg.norm(w_)
        w.append(w_)
    while point < num:
        v = np.random.randn(dp1)
        v /= np.linalg.norm(v)
        j += 1
        if sum([int(np.inner(w[i], v) > 0) for i in range(num_filters)]) <= (1-delta)/2.*num_filters:
            sphere_points[point] = v
            point += 1
    return sphere_points, j


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
        lossy_vec[index] = int(sgn(np.inner(vec, popcnt_vec)))
    return lossy_vec


def hamming_compare(lossy_vec1, lossy_vec2, k):
    """
    Takes an XOR of two SimHashes and checks the popcnt is above or below a
    threshold.

    :param lossy_vec1: the SimHash of the first point on the sphere
    :param lossy_vec2: the SimHash of the second point on the sphere
    :param k: the acceptance/rejection threshold for popcnts
    :returns: ``True`` if the pair passed the filter, ``False`` otherwise
    """
    assert len(lossy_vec1) == len(lossy_vec2), "Different dimension sketches!"
    n = len(lossy_vec1)
    hamming_xor = sum(np.bitwise_xor(lossy_vec1.astype(int),
                                     lossy_vec2.astype(int)))
    if hamming_xor < k or hamming_xor > n - k:
        return True
    return False


def gauss_test(dp1, num):
    """
    Quick function to generate experimental results for the proportion of
    i.i.d. vectors on the unit sphere ``S^{d} ⊂ R^{d+1}`` which are
    and are not Gauss reduced.

    :param dp1: the dimension of real space, i.e. `d+1`
    :param num: the number of vectors to sample and test
    :returns: the proportion of vector pairs which are Gauss reduced
    """
    gr = 0
    tot = 1./2. * num * (num - 1)
    vector = uniform_iid_sphere(dp1, num)
    for i, u in vector.items():
        for j, v in vector.items():
            if i <= j:
                continue
            gr += 1*gauss_reduced(u, v)
    return float(gr)/tot


def filter_test(dp1, num, n, k, spherical_code=False):
    """
    Quick function to generate experimental results for the proportion of
    i.i.d. vectors on the unit sphere S^{d} \subset R^{d+1} which do and
    do not pass the filter with n tests and threshold k.

    :param dp1: the dimension of real space
    :param num: the number of vectors to sample and test
    :param n: the number of vectors with which to make the SimHash
    :param k: the acceptance/rejection threshold for popcnts
    :param spherical_code: if ``False`` popcnt vectors come from the unit sphere,
                        else they mimic the behaviour of G6K

    :returns: the proportion of vector pairs passing the filter

    """
    pf = 0
    tot = 1./2. * num * (num - 1)
    popcnt = uniform_iid_sphere(dp1, n, spherical_code=spherical_code)
    vector = uniform_iid_sphere(dp1, num)
    for index, vec in vector.items():
        vector[index] = [vec, lossy_sketch(vec, popcnt)]
    for index1, vecs1 in vector.items():
        for index2, vecs2 in vector.items():
            if index1 >= index2:
                continue
            pf += hamming_compare(vecs1[1], vecs2[1], k)
    return float(pf)/tot


def filter_wrapper(dims, num, popcnt_nums, thresholds):
    """
    Quick check of accuracy of filter proportion estimates for lists of dims,
    popcnt_nums and thresholds to try. Prints basic stats.

    :param dims: a list of dp1 to be tried
    :param num: the number of vectors to sample and test
    :param popcnt_nums: a list of n to be tried
    :param thresholds: a list of thresholds to be tried
    """
    try:
        iter(dims)
    except TypeError:
        dims = [dims]
    try:
        iter(popcnt_nums)
    except TypeError:
        popcnt_nums = [popcnt_nums]
    try:
        iter(thresholds)
    except TypeError:
        thresholds = [thresholds]
    for dp1 in dims:
        for n in popcnt_nums:
            for k in thresholds:
                print
                print("dp1 %d, n %d, k %d"%(dp1, n, k))
                pf = filter_test(dp1, num, n, k)
                est = 2*pff(dp1, n, k)
                print("(exp, est): (%.6f, %.6f)"%(pf, est))
                print("="*25)


def gauss_wrapper(dims, num):
    """
    Quick check of accuracy of reduced proportion estimates for lists of dims.
    Prints basic stats.

    :param dims: a list of dp1 to be tried
    :param num: the number of vectors to sample and test
    """
    try:
        iter(dims)
    except TypeError:
        dims = [dims]
    for dp1 in dims:
        print
        print("dp1: %d"%dp1)
        gr = gauss_test(dp1, num)
        est = 1-2*ngrf(dp1)
        print("(exp, est): (%.6f, %.6f)"%(gr, est))
        print("="*25)
