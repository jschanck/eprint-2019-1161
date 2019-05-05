#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import cPickle
from sage.all import cached_function

from argparse import Namespace
from collections import OrderedDict
from mpmath import mp
from numpy import pi
from popcnt import estimate
# make sure this is the same as in popcnt.py
mp.prec = 212


def estimate_wrapper(d, n, k, efficient=False):
    """
    For a given real dimension, number of popcnt test vectors and a k,
    determines all useful probabilities.

    :param d: the dimension of real space
    :param n: the number of vectors with which to make the SimHash
    :param k: the acceptance/rejection k for popcnts
    :param efficient: if ``True`` only compute probs needed for (3)
    :returns: an OrderedDict with keys the names of a given probability
    """
    estimates = OrderedDict()

    gr = estimate(d, n, k, int_l=pi/3, int_u=(2*pi)/3, use_filt=False)
    estimates['gr'] = gr
    pf = estimate(d, n, k)
    estimates['pf'] = pf
    ngr_pf = mp.mpf('2')*estimate(d, n, k, int_u=pi/3)
    estimates['ngr_pf'] = ngr_pf

    if not efficient:
        gr_pf = estimate(d, n, k, int_l=pi/3, int_u=(2*pi)/3)
        estimates['gr_pf'] = gr_pf
        gr_npf = estimate(d, n, k, int_l=pi/3, int_u=(2*pi)/3, pass_filt=False)
        estimates['gr_npf'] = gr_npf
        ngr_npf = mp.mpf('2')*estimate(d, n, k, int_u=pi/3, pass_filt=False)
        estimates['ngr_npf'] = ngr_npf

    return estimates


def pretty_probs(estimates):
    """
    Takes an OrderedDict as returned by estimate_wrapper and prints the
    estimates probabilities.

    :param estimates: an OrderedDict of probabilities, from estimate_wrapper
    """
    for key, value in estimates.items():
        if key in ['gr', 'pf'] and value in [0, 1]:
            print key, 'has value', value
            print 'setting to', value, '\pm epsilon, so ignore relevant rows\n'
            if value == 0:
                estimates[key] += 0.1
            else:
                estimates[key] -= 0.1

    print 'gr:\t\t', estimates['gr']
    print 'ngr:\t\t', mp.mpf('1') - estimates['gr']
    print 'pf:\t\t', estimates['pf']
    print 'npf:\t\t', mp.mpf('1') - estimates['pf']
    print 'gr_pf:\t\t', estimates['gr_pf']
    print 'gr_npf:\t\t', estimates['gr_npf']
    print 'ngr_pf:\t\t', estimates['ngr_pf']
    print 'ngr_npf:\t', estimates['ngr_npf']
    print 'gr|pf:\t\t', estimates['gr_pf']/estimates['pf']
    print 'gr|npf:\t\t', estimates['gr_npf']/(mp.mpf('1') - estimates['pf'])
    print 'pf|gr:\t\t', estimates['gr_pf']/estimates['gr']
    print 'pf|ngr:\t\t', estimates['ngr_pf']/(mp.mpf('1') - estimates['gr'])
    print 'ngr|pf:\t\t', estimates['ngr_pf']/estimates['pf']
    print 'ngr|npf:\t', estimates['ngr_npf']/(mp.mpf('1') - estimates['pf'])
    print 'npf|gr:\t\t', estimates['gr_npf']/estimates['gr']
    print 'npf|ngr:\t', estimates['ngr_npf']/(mp.mpf('1') - estimates['gr'])
    print


def create_estimates(d, n=256, save=True, restrict=False, efficient=False):
    """
    Calculate estimates for all useful probabilities for a given dimension
    and a maximum allowed number of popcnt vectors.

    :param d: the dimension of real space
    :param n: the number of popcnt vectors to consider for a SimHash
    :param save: if ``True`` saves output OrderedDict as a pickle, if ``False``
                 returns OrderedDict
    :param restrict: restrict ``k`` to the interesting cases
    :param efficient: if ``True`` only compute probs needed for (3)
    :returns: an OrderedDict with keys tuples: (d, n, k)
    """
    all_estimates = OrderedDict()

    if not restrict:
        K = range(1, int(n/2.))
    else:
        K = range(max(int(0.3125*n)-5, 1), min(int(0.3125*n)+5+1, int(n//2)))
    for k in K:
        print n, k
        key = (d, n, k)
        all_estimates[key] = estimate_wrapper(d, n, k, efficient=efficient)

    if save:
        filename = 'probabilities/' + str(d) + '_' + str(n)
        with open(filename, 'wb') as f:
            cPickle.dump(all_estimates, f, -1)
    else:
        return all_estimates


def load_estimates(d, n=256):
    """
    Loads, if one exists, a pickle of the OrderedDict of OrderedDicts output
    by create_estimates(d, n=n, save=save).

    :param d: the dimension of real space
    :param n: the number of popcnt vectors to consider for a SimHash
    :returns: an OrderedDict with keys tuples: (d, n, k)
    """
    filename = 'probabilities/' + str(d) + '_' + str(n)
    try:
        with open(filename, 'rb') as f:
            all_estimates = cPickle.load(f)
        return all_estimates
    except IOError:
        raise NotImplementedError("No popcount parameters found")


@cached_function
def load_estimate(d, n, k, compute=False):
    try:
        all_estimates = load_estimates(d, n)
    except NotImplementedError as e:
        if compute is False:
            raise e
        else:
            create_estimates(d, n, restrict=True, efficient=True)
            return load_estimate(d, n, k, False)
    try:
        return Namespace(**all_estimates[(d, n, k)])
    except KeyError:
        if not compute:
            raise NotImplementedError("No such popcount parameters computed yet.")
        else:
            create_estimates(d, n, efficient=True)
            return load_estimate(d, n, k, False)


def maximise_optimising_func(d, f=None, n=256, verbose=False):
    """
    Apply a function f to all parameter choices (and their associated
    probabilities) for popcnts in a given dimension to optimise the popcnt
    parameters.

    :param d: the dimension of real space
    :param f: the function to maximise, if ``None`` uses default optimisation
              function ``optimisation`` defined below
    :param n: the number of popcnt vectors to consider for a SimHash
    :param verbose: if ``True`` print probabilities for each improving tuple
    :returns: a tuple (maximum of f, tuple of parameters that achieve it)
    """
    if f is None:
        f = optimisation
    all_estimates = load_estimates(d, n=n)
    if all_estimates is None:
        return
    solution_value = None
    solution_key = None
    for key, estimates in all_estimates.items():
        probs = Namespace(**estimates)
        new_value = f(probs.gr, probs.pf, probs.gr_pf, probs.gr_npf,
                      probs.ngr_pf, probs.ngr_npf)
        if new_value > solution_value or solution_value is None:
            solution_value = new_value
            solution_key = key
            if verbose:
                print solution_value, solution_key
                pretty_probs(estimates)
    return mp.mpf('1')/solution_value, solution_key


def grover_iterations(d, n, k, compute_probs=False):
    """
    A function for when you just want the number of Grover iterations required
    for a certain triple (d, n, k)

    :param d: the dimension of real space
    :param n: the number of popcnt vectors to consider for a SimHash
    :param k: the acceptance/rejection k for popcnts
    :returns: the calculated number of Grover iterations required
    """
    probs = load_estimate(d, n, k, compute_probs)
    inverse_giterations = optimisation(probs.gr, probs.pf, probs.gr_pf,
                                       probs.gr_npf, probs.ngr_pf,
                                       probs.ngr_npf)
    return mp.mpf('1')/inverse_giterations


def optimisation(gr, pf, gr_pf, gr_npf, ngr_pf, ngr_npf):
    # TODO: better name? This doesn't optimise anything
    exp1 = mp.fraction(1, 2)
    exp2 = mp.mpf('3')
    return ((ngr_pf**exp2)/((mp.mpf('1') - gr)*pf))**exp1


def _bulk_estimates_core(args):
    d, n = args
    try:
        return load_estimates(d, n)
    except NotImplementedError:
        return create_estimates(d, n, restrict=True)


def bulk_estimates(D, N=(128, 256, 512, 1024, 2048, 4096), ncores=1):
    """

    :param D:  list of dimension to consider
    :param N:  list of pocount counts
    :param ncores: number of cores to use

    """
    from multiprocessing import Pool

    jobs = []
    for d in D:
        for n in N:
            jobs.append((d, n))

    return list(Pool(ncores).imap_unordered(_bulk_estimates_core, jobs))
