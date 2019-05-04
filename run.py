#!/usr/bin/env sage
# -*- coding: utf-8 -*-
try:
    import click
except ImportError:
    print("Click not found. Run 'sage -pip install click'.")
    exit(1)

import logging
import numpy as np
from mpmath import mp
from collections import OrderedDict

from popcnt import uniform_iid_sphere, biased_sphere, lossy_sketch, gauss_reduced, hamming_compare, estimate

logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S %Z')
logger = logging.getLogger(__name__)


@click.command()
@click.argument("dim", type=int)
@click.argument("num", type=int)
@click.option("-n", help="Length of popcount hash", type=int, default=128)
@click.option("-k", help="Acceptance threshold, defaults to n/3", type=int, default=-1)
@click.option("--delta", help="If not 0, then prefilter points", default=0.0, type=float)
@click.option("--spherical-code/--no-spherical-codes", help="Select popcoutnt vectors as in G6K", default=False)
@click.option("--estimates/--no-estimates", help="Give integration based estimates.", default=True)
@click.option("--prec", help="Floating point precision", default=212, type=int)
@click.option("--seed", help="Random seed", default=None, type=int)
def run(dim, num, n, k, delta, spherical_code, estimates, prec, seed):
    """
    Detailed stats output for all relevant probabilities given real dimension dim, number of vectors
    to trial num, number of popcnt vectors popcnt_num, popcnt threshold threshold and a switch
    whether the popcnt vectors should mimic G6K.
    """

    mp.prec = prec
    np.random.seed(seed)
    if spherical_code or not estimates:
        print("Ignore integration based estimates!")

    # get dictionaries of points to be tested and used to form SimHashes
    popcnt = uniform_iid_sphere(dim, n, spherical_code=spherical_code)

    if delta > 0:
        L, tot_sampled = biased_sphere(dim, num, n, delta)
    else:
        L = uniform_iid_sphere(dim, num)

    # create SimHashes
    for index, vec in L.items():
        L[index] = [vec, lossy_sketch(vec, popcnt)]

    if k == -1:
        k =n//3
    dim = mp.mpf(dim)
    num = mp.mpf(num)
    n = mp.mpf(n)
    k = mp.mpf(k)

    # initialise counters
    # (n)gr = (not) Gauss reduced, (n)pf = (did not) pass filter
    gr_pf = mp.mpf('0')
    gr_npf = mp.mpf('0')
    ngr_pf = mp.mpf('0')
    ngr_npf = mp.mpf('0')
    tot = mp.fraction(1, 2) * num * (num - mp.mpf('1'))

    # compare pairwise different vectors and their SimHashes
    for index1, vecs1 in L.items():
        for index2, vecs2 in L.items():
            if index1 >= index2:
                continue
            gauss_check = gauss_reduced(vecs1[0], vecs2[0])
            lossy_check = hamming_compare(vecs1[1], vecs2[1], k)

            # information about filter pass/fail and Gauss reduction for pair
            if gauss_check and lossy_check:
                gr_pf += mp.mpf('1')
            if gauss_check and not lossy_check:
                gr_npf += mp.mpf('1')
            if not gauss_check and lossy_check:
                ngr_pf += mp.mpf('1')
            if not gauss_check and not lossy_check:
                ngr_npf += mp.mpf('1')

    gr = gr_pf + gr_npf
    pf = gr_pf + ngr_pf

    if gr == 0:
        logger.warn("None Gauss reduced, setting gr to 1, this should not happen")
        gr = mp.mpf('1')
    elif gr == tot:
        logger.warn("All Gauss reduced, setting gr to tot - 1, pick a larger num")
        gr = tot - mp.mpf('1')

    if pf == 0:
        logger.warn("Nothing passed filter, setting pf to 1, pick larger k")
        pf = mp.mpf('1')
    elif pf == tot:
        logger.warn("Everything passed filter, setting pf to tot - 1, pick smaller k")
        pf = tot - mp.mpf('1')

    # collect all stats, of form [absolute value, tot_ratio, est]
    stats = OrderedDict()

    if estimates:
        est_gr = estimate(dim, n, k, int_l=mp.pi/mp.mpf('3'),
                          int_u=(2*mp.pi)/mp.mpf('3'), use_filt=False)
        est_pf = estimate(dim, n, k)
        est_gr_pf = estimate(dim, n, k,
                             int_l=mp.pi/mp.mpf('3'),
                             int_u=(2*mp.pi)/mp.mpf('3'))
        est_gr_npf = estimate(dim, n, k,
                              int_l=mp.pi/mp.mpf('3'),
                              int_u=(2*mp.pi)/mp.mpf('3'), pass_filt=False)
        est_ngr_pf = 2*estimate(dim, n, k,
                                int_u=mp.pi/mp.mpf('3'))
        est_ngr_npf = 2*estimate(dim, n, k,
                                 int_u=mp.pi/mp.mpf('3'), pass_filt=False)
    else:
        est_gr = 2
        est_pf = 2
        est_gr_pf = 2
        est_gr_npf = 2
        est_ngr_pf = 2
        est_ngr_npf = 2

    stats['gr'] = [int(gr), gr/tot, est_gr]
    stats['ngr'] = [int(tot-gr), (tot-gr)/tot, mp.mpf('1') - est_gr]
    stats['pf'] = [int(pf), pf/tot, est_pf]
    stats['npf'] = [int(tot-pf), (tot-pf)/tot, mp.mpf('1') - est_pf]
    stats['gr_pf'] = [int(gr_pf), gr_pf/tot, est_gr_pf]
    stats['gr_npf'] = [int(gr_npf), gr_npf/tot, est_gr_npf]
    stats['ngr_pf'] = [int(ngr_pf), ngr_pf/tot, est_ngr_pf]
    stats['ngr_npf'] = [int(ngr_npf), ngr_npf/tot, est_ngr_npf]

    # conditional probabilities, e.g. the first is gr given that pf
    space = "-"*10
    stats['gr|pf'] = [space, gr_pf/pf, est_gr_pf/est_pf]
    stats['gr|npf'] = [space, gr_npf/(tot-pf), est_gr_npf/(mp.mpf('1')-est_pf)]
    stats['pf|gr'] = [space, gr_pf/gr, est_gr_pf/est_gr]
    stats['pf|ngr'] = [space, ngr_pf/(tot-gr), est_ngr_pf/(mp.mpf('1')-est_gr)]
    stats['ngr|pf'] = [space, ngr_pf/pf, est_ngr_pf/est_pf]
    stats['ngr|npf'] = [space, ngr_npf/(tot-pf), est_ngr_npf/(mp.mpf('1')-est_pf)] # noqa
    stats['npf|gr'] = [space, gr_npf/gr, est_gr_npf/est_gr]
    stats['npf|ngr'] = [space, ngr_npf/(tot-gr), est_ngr_npf/(mp.mpf('1')-est_gr)] # noqa

    for key, value in stats.items():
        print "{key:10s}:: abs: {abs:>10s}, exp: {exp:11.9f}, est: {est:11.9f}".format(key=key,
                                                                                       abs=str(value[0]),
                                                                                       exp=float(value[1]),
                                                                                       est=float(value[2]))

    if stats["ngr_pf"][1]:
        print "   Eq (3) :: exp:", int(round(stats["ngr"][1] * stats["pf"][1] / stats["ngr_pf"][1]**3))


if __name__ == '__main__':
    run()
