#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import cPickle

from argparse import Namespace
from collections import OrderedDict
from mpmath import mp
from numpy import pi
from popcnt import estimate
# make sure this is the same as in popcnt.py
mp.prec = 212


def estimate_wrapper(dim, popcnt_num, threshold):
    """
    For a given real dimension, number of popcnt test vectors and a threshold,
    determines all useful probabilities.

    :param dim: the dimension of real space
    :param popcnt_num: the number of vectors with which to make the SimHash
    :param threshold: the acceptance/rejection threshold for popcnts
    :returns: an OrderedDict with keys the names of a given probability
    """
    estimates = OrderedDict()

    gr = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3,
                  use_filt=False)
    estimates['gr'] = gr
    pf = estimate(dim, popcnt_num, threshold)
    estimates['pf'] = pf
    gr_pf = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3)
    estimates['gr_pf'] = gr_pf
    gr_npf = estimate(dim, popcnt_num, threshold, int_l=pi/3, int_u=(2*pi)/3,
                      pass_filt=False)
    estimates['gr_npf'] = gr_npf
    ngr_pf = mp.mpf('2')*estimate(dim, popcnt_num, threshold, int_u=pi/3)
    estimates['ngr_pf'] = ngr_pf
    ngr_npf = mp.mpf('2')*estimate(dim, popcnt_num, threshold, int_u=pi/3,
                                   pass_filt=False)
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


def create_estimates(dim, popcnt_num=256, save=True):
    """
    Calculate estimates for all useful probabilities for a given dimension
    and a maximum allowed number of popcnt vectors.

    :param dim: the dimension of real space
    :param popcnt_num: the number of popcnt vectors to consider for a SimHash
    :param save: if ``True`` saves output OrderedDict as a pickle, if ``False``
                 returns OrderedDict
    :returns: an OrderedDict with keys tuples: (dim, popcnt_num, threshold)
    """
    all_estimates = OrderedDict()
    for threshold in range(1, int(popcnt_num/2.)):
        print popcnt_num, threshold
        key = (dim, popcnt_num, threshold)
        all_estimates[key] = estimate_wrapper(dim, popcnt_num, threshold)

    if save:
        filename = 'probabilities/' + str(dim) + '_' + str(popcnt_num)
        with open(filename, 'wb') as f:
            cPickle.dump(all_estimates, f, -1)
    else:
        return all_estimates


def load_estimates(dim, popcnt_num=256):
    """
    Loads, if one exists, a pickle of the OrderedDict of OrderedDicts output
    by create_estimates(dim, popcnt_num=popcnt_num, save=save).

    :param dim: the dimension of real space
    :param popcnt_num: the number of popcnt vectors to consider for a SimHash
    :returns: an OrderedDict with keys tuples: (dim, popcnt_num, threshold)
    """
    filename = 'probabilities/' + str(dim) + '_' + str(popcnt_num)
    try:
        with open(filename, 'rb') as f:
            all_estimates = cPickle.load(f)
        return all_estimates
    except IOError:
        print 'Please run create_estimates(%d, %d)' % (dim, popcnt_num)


def maximise_optimising_func(dim, f=None, popcnt_num=256, verbose=False):
    """
    Apply a function f to all parameter choices (and their associated
    probabilities) for popcnts in a given dimension to optimise the popcnt
    parameters.

    :param dim: the dimension of real space
    :param f: the function to maximise, if ``None`` uses default optimisation
              function ``optimisation`` defined below
    :param popcnt_num: the number of popcnt vectors to consider for a SimHash
    :param verbose: if ``True`` print probabilities for each improving tuple
    :returns: a tuple (maximum of f, tuple of parameters that achieve it)
    """
    if f is None:
        f = optimisation
    all_estimates = load_estimates(dim, popcnt_num=popcnt_num)
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


def grover_iterations(dim, popcnt_num, threshold):
    """
    A function for when you just want the number of Grover iterations required
    for a certain triple (dim, popcnt_num, threshold)

    :param dim: the dimension of real space
    :param popcnt_num: the number of popcnt vectors to consider for a SimHash
    :param threshold: the acceptance/rejection threshold for popcnts
    :returns: the calculated number of Grover iterations requiredd
    """
    all_estimates = load_estimates(dim, popcnt_num=popcnt_num)
    if all_estimates is None:
        return
    try:
        estimates = all_estimates[(dim, popcnt_num, threshold)]
    except KeyError:
        print "No such popcount parameters exist"
        return
    probs = Namespace(**estimates)
    inverse_giterations = optimisation(probs.gr, probs.pf, probs.gr_pf,
                                       probs.gr_npf, probs.ngr_pf,
                                       probs.ngr_npf)
    return mp.mpf('1')/inverse_giterations


def optimisation(gr, pf, gr_pf, gr_npf, ngr_pf, ngr_npf):
    exp1 = mp.fraction(1, 2)
    exp2 = mp.mpf('3')
    return ((ngr_pf**exp2)/((mp.mpf('1') - gr)*pf))**exp1


"""
def optimisation(gr, pf, gr_pf, gr_npf, ngr_pf, ngr_npf):
    exp1 = mp.fraction(-3, 4)
    exp2 = mp.mpf('-1')
    return ((ngr_pf/(mp.mpf('1')-gr))**(exp1)*(ngr_pf/pf)**(exp2))**(exp2)


def optimisation(gr, pf, gr_pf, gr_npf, ngr_pf, ngr_npf):
    # we want pf|ngr >= c, then to maximise ngr_pf/pf
    if ngr_pf/(mp.mpf('1') - gr) >= mp.fraction(1, 10):
        return ngr_pf/pf
    else:
        return None
"""


def _bulk_estimates_core(args):
    d, c = args
    return create_estimates(d, int(round(c*d)))


def bulk_estimates(D, C=(0.5, 1.0, 2.0, 4.0), ncores=1):
    """

    :param D:  list of dimension to consider
    :param C:  list of `c` s.t. `n=c⋅d`
    :param ncores: number of cores to use

    """
    from multiprocessing import Pool

    jobs = []
    for d in D:
        for c in C:
            jobs.append((d, c))

    return list(Pool(ncores).imap_unordered(_bulk_estimates_core, jobs))
