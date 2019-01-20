#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from collections import OrderedDict
from gmpy import comb
import numpy as np
import sys


def sign(value):
    """
    Utility function for computing the sign of real number.

    :param value: a float or int
    :returns: ``True`` if value is >= 0, else ``False``.
    """
    if value >= 0:
        return 1
    else:
        return 0


def uniform_iid_sphere(dim, num):
    """
    Samples points uniformly on the unit sphere in dimension dim.
    It samples from dim many zero centred Gaussians and normalises the vector.

    :param dim: the dimension of the unit sphere
    :param num: number of points to be sampled uniformly and i.i.d
    :returns: an OrderedDict with keys which enumerate the points as values.
    """
    sphere_points = OrderedDict()
    for point in range(num):
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

    :param vec: the point on the sphere to make a SimHash of.
    :param popcnt_dict: the fixed vectors with which to make the SimHash.
    :returns: an integer numpy array with values in {0, 1}, a SimHash.
    """
    lossy_vec = np.zeros(len(popcnt_dict))
    for index, popcnt_vec in popcnt_dict.items():
        lossy_vec[index] = int(sign(np.inner(vec, popcnt_vec)))
    return lossy_vec


def hamming_compare(lossy_vec1, lossy_vec2, popcnt_num, threshold):
    """
    Takes an XOR of two SimHashes and checks the popcnt is above or below a
    threshold.

    :param lossy_vec1: the SimHash of the first point on the sphere.
    :param lossy_vec2: the SimHash of the second point on the sphere.
    :param popcnt_num: the number of fixed vectors used to make SimHashes.
    :param threshold: the acceptance/rejection threshold for popcnts.
    :returns: ``True`` if the pair need further testing, ``False`` otherwise.
    """
    hamming_xor = sum(np.bitwise_xor(lossy_vec1.astype(int),
                                     lossy_vec2.astype(int)))
    if hamming_xor <= threshold or hamming_xor >= popcnt_num - threshold:
        return True
    else:
        return False


def main():
    dim, num, popcnt_num, threshold = sys.argv[1:]

    dim = int(dim)
    num = int(num)
    popcnt_num = int(popcnt_num)
    threshold = int(threshold)

    # initialise popcnt counters
    popcnt_suc = 0
    popcnt_tot = 1./2. * num * (num - 1)

    # get dictionaries of points to be tested, used to form SimHashes and an
    # empty dictionary for the SimHashes
    popcnt_dict = uniform_iid_sphere(dim, popcnt_num)
    vec_dict = uniform_iid_sphere(dim, num)
    lossy_dict = OrderedDict()

    # create SimHashes
    for index, vec in vec_dict.items():
        lossy_vec = lossy_sketch(vec, popcnt_dict)
        lossy_dict[index] = lossy_vec

    # compare pairwise different SimHashes and count those for further tests.
    for index1, lossy_vec1 in lossy_dict.items():
        for index2, lossy_vec2 in lossy_dict.items():
            if index1 >= index2:
                continue
            check = hamming_compare(lossy_vec1, lossy_vec2, popcnt_num,
                                    threshold)
            popcnt_suc += 1*check

    ratio = popcnt_suc/float(popcnt_tot)
    print ratio

    # calculate my estimate for number of further tests.
    est1 = (1./2.)**(popcnt_num - 1)
    est2 = sum([int(comb(popcnt_num, i)) for i in range(0, threshold + 1)])
    print est1 * est2


def dim_inc_test():
    # test that as dim -> popcnt_num my estimate gets more accurate
    popcnt_num = 128
    threshold = 32
    for dim in range(40, 129):
        print dim
        popcnt_suc = 0
        popcnt_tot = 500 * 999

        popcnt_dict = uniform_iid_sphere(dim, 128)
        vec_dict = uniform_iid_sphere(dim, 1000)
        lossy_dict = OrderedDict()

        for index, vec in vec_dict.items():
            lossy_vec = lossy_sketch(vec, popcnt_dict)
            lossy_dict[index] = lossy_vec

        for index1, lossy_vec1 in lossy_dict.items():
            for index2, lossy_vec2 in lossy_dict.items():
                if index1 >= index2:
                    continue
                check = hamming_compare(lossy_vec1, lossy_vec2, popcnt_num,
                                        threshold)
                popcnt_suc += 1*check

        ratio = popcnt_suc/float(popcnt_tot)
        print ratio

        est1 = (1./2.)**(popcnt_num - 1)
        est2 = sum([int(comb(popcnt_num, i)) for i in range(0, threshold + 1)])
        print est1 * est2

        print ratio/float(est1 * est2)
        print


if __name__ == '__main__':
    main()
