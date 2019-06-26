#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mpmath import mp
from collections import namedtuple
from utils import load_probabilities
from config import MagicConstants

LogicalCosts = namedtuple("LogicalCosts",
                          ("label", "params",
                           "cnots_count", "toffoli_count",
                           "t_count", "t_depth", "t_width"))


def _preproc_params(L, n, k):
    """
    Normalise inputs and check for consistency.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    if not 0 <= k <= n:
        raise ValueError("k=%d not in range 0 ... %d"%(k, n))

    if mp.log(n, 2)%1 != 0:
        raise ValueError("n=%d is not a power of two"%n)

    index_wires = mp.ceil(mp.log(L, 2))
    if index_wires < 4:
        raise ValueError("diffusion operator poorly defined, log|L|=%d too small."%index_wires)

    # a useful value for computation that follows the writeup
    return L, n, k, index_wires


def ellf(n):
    return mp.log(n, 2) + 1


def popcount_costf(L, n, k):
    """
    Logical cost of running popcount filter once.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """

    L, n, k, index_wires = _preproc_params(L, n, k)

    # TODO: magic constants
    OR_CNOTs = 2
    OR_Tofs = 1

    # number of ORs required to test whether popcount is less than 2^t, some t in {0, 1, 2, ..., l -
    # 1}, is l - t - 1, i.e. more for smaller t. we say k in [ 2^t + 1 , 2^(t + 1) ] costs the same
    # number of ORs

    t = mp.ceil(mp.log(k, 2))
    ORs = ellf(n) - t - 1

    def i_bit_adder_CNOTs(i):
        # to achieve the Toffoli counts for i_bits_adder_Tofs() below we diverge from the expected
        # number of CNOTs for 1 bit adders
        if i == 1:
            return 6
        else:
            return 5*i - 3

    def i_bit_adder_Tofs(i):
        # these Toffoli counts for 1 and 2 bit adders can be achieved using (some of) the
        # optimisations in Cuccarro et al.
        return 2*i - 1

    def i_bit_adder_T_depth(i):
        # Each i bit adder has 2i - 1 Toffoli gates (sequentially) so using T depth 3 per Toffoli
        # gives 6i - 3 T depth for an i bit adder
        return 6 * i - 3

    adder_cnots    = n*sum([i_bit_adder_CNOTs(i)/float(2**i) for i in range(1, ellf(n))]) # noqa
    popcount_cnots = n + OR_CNOTs*ORs + adder_cnots

    adder_tofs      = n*sum([i_bit_adder_Tofs(i)/float(2**i) for i in range(1, ellf(n))]) # noqa
    popcount_tofs   = OR_Tofs*ORs + adder_tofs

    # all i bit adders are in parallel and we use 1, ..., log_2(n) bit adders
    adder_t_depth   = sum([i_bit_adder_T_depth(i) for i in range(1, ellf(n))])
    # we have ceil(ell - t) OR depth, 1 Tof therefore 3 T-depth each
    OR_t_depth = 3 * mp.ceil(mp.log(ellf(n) - t, 2))
    popcount_t_depth = adder_t_depth + OR_t_depth

    popcount_t_count = MagicConstants.t_div_toffoli * (popcount_tofs)

    qc = LogicalCosts(label="popcount",
                      params=(L, n, k),
                      cnots_count=popcount_cnots,
                      toffoli_count=popcount_tofs,
                      t_count=popcount_t_count,
                      t_depth=popcount_t_depth,
                      t_width=popcount_t_count/popcount_t_depth)
    return qc


def oracle_costf(L, n, k):
    """
    Logical cost of calling Grover oracle once.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    L, n, k, index_wires = _preproc_params(L, n, k)
    popcount_cost = popcount_costf(L, n, k)

    # a l-controlled NOT takes (32l - 84)T
    # the diffusion operator in our circuit is (index_wires - 1)-controlled NOT
    # TODO: magic constants
    diffusion_t_count = 32 * (index_wires - 1) - 84

    # We currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    diffusion_t_depth = diffusion_t_count

    # TODO: could also handle CNOTs and Toffoli
    t_count = 2*popcount_cost.t_count + diffusion_t_count
    t_depth = 2*popcount_cost.t_depth + diffusion_t_depth
    t_width = t_count/t_depth

    return LogicalCosts(label="oracle",
                        params=(L, n, k),
                        cnots_count=None,
                        toffoli_count=None,
                        t_count=t_count, t_depth=t_depth, t_width=t_width)


def searchf(L, n, k):
    """
    Logical cost of running quantum quadratic Nearest Neighbor Search.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    # TODO: magic constants for amplitude amplification?
    L, n, k, index_wires = _preproc_params(L, n, k)

    oracle_cost = oracle_costf(L, n, k)
    oracle_calls = mp.sqrt(L)

    t_count = oracle_calls * oracle_cost.t_count
    t_depth = oracle_calls * oracle_cost.t_depth
    t_width = oracle_cost.t_width

    return LogicalCosts(label="search",
                        params=(L, n, k),
                        cnots_count=None,
                        toffoli_count=None,
                        t_count=t_count, t_depth=t_depth, t_width=t_width)


def all_pairs(d, n, k=None, epsilon=0.01):
    k = k if k else int(MagicConstants.k_div_n * n)
    pr = load_probabilities(d, n, k)
    epsilon = mp.mpf(epsilon)

    L = (1+epsilon) * 2/((1-pr.eta)*pr.ngr)
    L, n, k, index_wires = _preproc_params(L, n, k)

    calls = int(mp.ceil(2*(1+epsilon)/(1-pr.eta) * L))
    size = ((1-pr.eta)/(1+epsilon))**2 * L/2
    search_cost = searchf(size, n, k)

    # TODO: What do we want to return here?
    return calls, search_cost, calls * search_cost.t_count
