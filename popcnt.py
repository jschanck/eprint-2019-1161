# -*- coding: utf-8 -*-

import numpy as np
from mpmath import mp
from collections import OrderedDict


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


def uniform_iid_sphere(dim, num, spherical_code=False):
    """
    Samples ``num`` points uniformly on the unit sphere in ``R^{dim}``, i.e. S^{d - 1}.
    It samples from dim many zero centred Gaussians and normalises the vector.

    :param dim: the dimension of real space
    :param num: number of points to be sampled uniformly and i.i.d
    :param spherical_code: if ``False`` popcnt vectors come from the unit sphere,
                        else they mimic the behaviour of G6K
    :returns: an OrderedDict with keys which enumerate the points as values
    """
    sphere_points = OrderedDict()

    if not spherical_code:
        for point in range(num):
            # good randomness seems quite pertinent
            vec = np.random.randn(dim)
            vec /= np.linalg.norm(vec)
            sphere_points[point] = vec
    else:
        with open("spherical_coding/sc_{dim}_{num}.def".format(dim=dim, num=num)) as f:
            for point, line in enumerate(f.readlines()): # noqa
                idx = map(int, line.split(" "))
                vec = [0]*dim
                for i in idx[:3]:
                    vec[i] = 1
                for i in idx[3:]:
                    vec[i] = -1
                sphere_points[point] = np.array(vec)
    return sphere_points


def biased_sphere(d, num, n, delta):
    """

    :param d: the dimension of real space
    :param num: number of points to be sampled uniformly and i.i.d
    :param delta:
    :param n:
    :returns: an OrderedDict with keys which enumerate the points as values

    """
    sphere_points = OrderedDict()
    point = 0

    w = []
    j = 0
    for i in range(n):
        w_ = np.random.randn(d)
        w_ /= np.linalg.norm(w_)
        w.append(w_)
    while point < num:
        v = np.random.randn(d)
        v /= np.linalg.norm(v)
        j += 1
        if sum([int(np.inner(w[i], v) > 0) for i in range(n)]) <= (1-delta)/2.*n:
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
        lossy_vec[index] = int(sign(np.inner(vec, popcnt_vec)))
    return lossy_vec


def hamming_compare(lossy_vec1, lossy_vec2, threshold):
    """
    Takes an XOR of two SimHashes and checks the popcnt is above or below a
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
    if hamming_xor < threshold or hamming_xor > popcnt_num - threshold:
        return True
    return False


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


def split_interval(int_l, int_u, slices):
    """
    Splits closed interval [intl_l, int_u] into ``slices`` number of slices of
    equal size

    :param int_l: the lower bound of the interval
    :param int_u: the upper bound of the interval
    :param slices: the number of slices of equal size to split interval into
    :returns: list of the values that define the slices
    """
    int_l = mp.mpf(int_l)
    int_u = mp.mpf(int_u)
    intervals = [int_l]
    for i in range(1, slices + 1):
        intervals += [int_l + mp.mpf(i)*(int_u - int_l)/mp.mpf(slices)]
    return intervals


def estimate(dim, popcnt_num, threshold, int_l=0, int_u=mp.pi, use_filt=True,
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
    d = mp.mpf(dim)
    n = mp.mpf(popcnt_num)
    k = mp.mpf(threshold)

    # if the integrations are not accurate, increase the intervals
    interval = split_interval(int_l, int_u, 1)

    if use_filt:
        if pass_filt:
            coeffs = [mp.binomial(n, i) for i in range(0, k)]
            coeffs += [mp.mpf('0')] * int(n - (2 * k) + 1)
            coeffs += [mp.binomial(n, i) for i in range(n - k + 1, n + 1)]
        else:
            coeffs = [mp.mpf('0')] * int(k)
            coeffs += [mp.binomial(n, i) for i in range(k, n - k + 1)]
            coeffs += [mp.mpf('0')] * int(k)

        prob = 0
        for i in range(n + 1):
            co = coeffs[i]
            i = mp.mpf(i)
            if co == 0:
                continue

            def f(x): return (mp.sin(x)**(d-mp.mpf('2')))*co*((x/mp.pi)**i)*((mp.mpf('1')-(x/mp.pi))**(n-i)) # noqa
            prob += mp.quad(f, interval, maxdegree=50000, error=True)[0]

    else:

        def f(x): return (mp.sin(x)**(d-mp.mpf('2')))
        prob = mp.quad(f, interval, maxdegree=50000, error=True)[0]

    def normaliser(dim):
        """
        The normalisation constant (dependent on dim) has a closed form!
        .. note:: we are interested in the relative surface area of
        (hyper)spheres on the surface of S^{dim - 1}, hence dim - 2.

        :param dim: the dimension of real space
        :returns: the normalisation constant for the integral estimates
        """
        d = mp.mpf(dim) - mp.mpf('2')
        norm = mp.mpf('1')
        if d % 2 == 0:
            for i in range(1, d + 1):
                if i % 2 == 0:
                    norm *= mp.mpf(i)**mp.mpf('-1')
                else:
                    norm *= mp.mpf(i)
            norm *= mp.pi
        else:
            for i in range(2, d + 1):
                if i % 2 == 0:
                    norm *= mp.mpf(i)
                else:
                    norm *= mp.mpf(i)**mp.mpf('-1')
            norm *= mp.mpf(2)
        return mp.mpf(1)/norm

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
        est = estimate(dim, 0, 0, int_l=mp.pi/mp.mpf('3'),
                       int_u=(2*mp.pi)/mp.mpf('3'), use_filt=False)
        print "(exp, est): (%.6f, %.6f)" % (gr, est)
        print "="*25


