#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import cPickle
from mpmath import mp
from optimise_popcnt import grover_iterations, load_estimate
from collections import OrderedDict


def _preproc_params(d, n, k, compute_probs=True, speculate=False):
    """
    Normalise inputs and check for consistency.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter
    :param compute_probs: compute probabilities if needed
    :param speculate: pretend popcount is perfect

    """

    # For a dimension d lattice using an NV like sieve we expect Oe(2^{0.2075*d}) lattice vectors.
    # We round this up to the nearest power of two to be able to use Hadamard gates to set up
    # Grover's algorithm.

    if not 0 <= k < n//2:
        raise ValueError("k (%d) not in range 0 ... n//2-1 (%d)"%(k, n))

    # determines width of the diffusion operator and accounts for list growth
    probs = load_estimate(d, n, k, compute=compute_probs)
    if not speculate:
        list_growth = (probs.ngr_pf/(mp.mpf('1') - probs.gr))**(-1./2.)
    else:
        list_growth = 1
    index_wires = mp.ceil(mp.log(list_growth * (2**(0.2075*d)), 2))
    if index_wires < 4:
        raise ValueError("diffusion operator poorly defined, d = %d too small."%d)

    return d, n, k, index_wires


def T_count_giteration(d, n, k):
    """
    T-count for one Grover iteration.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter

    """

    d, n, k, index_wires = _preproc_params(d, n, k)

    tof_count_adder = n * mp.fraction(7, 2) + mp.log(n, 2) - k
    # TODO: this can be negative
    # NOTE: now exact, before using upper bound that tends to almost 2x too many Ts
    # ell = mp.log(n, 2) + 1
    # tof_count_adder = n * sum([(2 * i - 1)/(2**i) for i in range(1, ell)]) + ell - k - 1
    # each Toffoli costs approx 7T
    T_count_adder = 7 * tof_count_adder

    # a l-controlled NOT takes (32l - 84)T
    # the diffusion operator in our circuit is (index_wires - 1)-controlled NOT
    T_count_diffusion = 32 * (index_wires - 1) - 84

    # we have adder and its inverse
    return 2 * T_count_adder + T_count_diffusion


def T_depth_giteration(d, n, k):
    """
    T-depth of one Grover iteration.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter

    """

    d, n, k, index_wires = _preproc_params(d, n, k)

    def T_depth_i_adder(i):
        """
        Each i bit adder has 2i - 1 Toffoli gates (sequentially) so using T
        depth 3 per Toffoli gives 6i - 3 T depth for an i bit adder
        """
        # TODO: this is not true for i \in {1, 2}
        return 6 * i - 3

    # all i bit adders are in parallel and we use 1, ..., log_2 n bit adders
    upper = int(mp.log(n, 2) + 1)
    T_depth_adder = sum([T_depth_i_adder(bits) for bits in range(1, upper)])

    # I currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    # TODO: is this assumption necessarily favourable always?
    T_depth_diffusion = 32 * (index_wires - 1) - 84
    return 2 * T_depth_adder + T_depth_diffusion


def T_average_width_giteration(d, n, k):
    """
    Take the floor (generous to sieving adversary) of the division of T_count
    and T_depth of the given circuit to determine how many T gates required
    on average T depth
    """
    return mp.floor(T_count_giteration(d, n, k)/float(T_depth_giteration(d, n, k)))


def fifteen_one(p_out, p_in, p_g=None, eps=None):
    """
    Find the distances required for Reed-Muller distillation of magic states
    """
    if p_g is None:
        p_g = p_in/mp.mpf('10')
    if eps is None:
        eps = mp.mpf('1')
    d, p = [], p_out
    while p <= p_in:
        d_i = None
        for d_i in range(1, 100):
            if 192*d_i*(100*p_g)**((d_i+1)/2.) < eps*p/(1+eps):
                break
        p = (p/(35*(1+eps)))**(1/3.)
        d.append(d_i)
    return d


def num_physical_qubits(distance):
    """
    Physical qubits per error corrected logical qubit for surface code with distance ``distance``

    See Appendix M of /Surface codes: Towards practical large-scale quantum computation/ by Austin G.
    Fowler, Matteo Mariantoni, John M.  Martinis and Andrew N.  Cleland

    """
    return 1.25 * 2.5 * ((2 * distance) ** 2)


def clifford_gates(d, n, k, total_giterations):
    """
    Calculate all the Clifford gates required in as many Grover iterations
    as the popcnt parameters require
    """
    d, n, k, index_wires = _preproc_params(d, n, k)

    # not currently counting NOTs in adder
    cnot_adder = n * mp.fraction(19, 2) + mp.mpf('2') * (mp.log(n, 2) - k)
    not_diffusion = 2 * (index_wires - 1)
    hadamard_diffusion = 2 * (index_wires - 1)
    z_diffusion = 2

    return 2 * cnot_adder + not_diffusion + hadamard_diffusion + z_diffusion


def distance_condition_clifford(p_in, num_clifford_gates):
    """
    For the Clifford gates in a Grover iteration, we need a distance as
    calculated below.
    """
    d = 1
    while True:
        if (80 * p_in)**((d + 1)/2.) < 1./num_clifford_gates:
            break
        d += 1
    return d


def wrapper(d, n, k=None, p_in=10.**(-4), p_g=10.**(-5), compute_probs=True, speculate=False):
    if k is None:
        best = None
        # NOTE: 5/16*n seems to be optimal
        for k in range(max(int(0.3125*n)-5, 1), min(int(0.3125*n)+5+1, int(n//2))):
            cur = wrapper(d, n, k, p_in=p_in, p_g=p_g, compute_probs=compute_probs, speculate=speculate)
            if best is None or cur < best:
                best = cur
        return best
    _, _, _, index_wires = _preproc_params(d, n, k,
                                           compute_probs=compute_probs,
                                           speculate=speculate)

    # we will eventually interpolate between non power of two n, sim for k
    assert(mp.log(n, 2)%1 ==0), "Not a power of two n!"
    # TODO: we permit non-power-of-two ks for now

    # calculating the total number of T gates for required error bound
    if not speculate:
        total_giterations = grover_iterations(d, n, k, compute_probs=compute_probs)
    else:
        total_giterations = mp.ceil(mp.pi/4*2**(0.2075/2*d))
    T_count_total = total_giterations * T_count_giteration(d, n, k)
    p_out = mp.mpf('1')/T_count_total

    p_in = mp.mpf(p_in)
    p_g = mp.mpf(p_g)

    # distances, and physical qubits per logical for the layers of distillation
    distances = fifteen_one(p_out, p_in, p_g=p_g)
    layers = len(distances)
    # NOTE: d_last is used for circuit with biggest logical footprint
    phys_qbits = [num_physical_qubits(distance) for distance in distances[::-1]]

    # physical qubits per layer, starting with topmost
    phys_qbits_layer = [16*(15**(layers-i))*phys_qbits[i-1] for i in range(1, layers + 1)] # noqa

    # total surface code cycles per magic state distillation (not pipelined)
    scc = 10 * sum(distances)

    # total number of physical/logical qubits for msd
    # total_distil_phys_qbits = max(phys_qbits_layer)
    total_distil_logi_qbits = 16 * (15 ** (layers - 1))

    # how many magic states can we pipeline at once?
    if layers == 1:
        parallel_msd = 1
    else:
        parallel_msd = mp.floor(max(float(phys_qbits_layer[0])/phys_qbits_layer[1], 1))

    # the average T gates per T depth
    T_average = T_average_width_giteration(d, n, k)

    # number of magic state distilleries required
    msds = mp.ceil(float(T_average)/parallel_msd)

    # logical qubits for Grover iteration = max(width of popcnt circuit, width of diffusion operator) + 1
    logi_qbits_giteration = max(3 * n + mp.log(n, 2) - k - 1, index_wires) + 1

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # distance required for Clifford gates, current ignoring Hadamards in setup
    # giteration_clifford_gates = clifford_gates(d, n, k, total_giterations)
    # clifford_distance = distance_condition_clifford(p_in, giteration_clifford_gates) # noqa

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # physical qubits for Grover iteration = width * f(clifford_distance)
    # phys_qbits_giteration = logi_qbits_giteration * num_physical_qubits(clifford_distance) # noqa

    # total number of logical qubits is
    total_logi_qbits = msds * total_distil_logi_qbits + logi_qbits_giteration
    # total number of surface code cycles for all the Grover iterations
    # NOTE: removed msds from total_scc because it should not affect depth
    total_scc = total_giterations * scc * T_depth_giteration(d, n, k)
    # total cost (ignoring Cliffords in error correction) is
    total_cost = total_logi_qbits * total_scc

    return float(total_cost), float(mp.log(total_cost, 2)), k


def _bulk_wrapper_core(args):
    d, n = args
    r = (d,) + wrapper(d, n) + wrapper(d, n, speculate=True)
    print(r)
    return r


def bulk_wrapper(D, N=(32, 64, 128, 256, 512), ncores=1):
    from multiprocessing import Pool

    jobs = []
    for n in N:
        for d in D:
            jobs.append((d, n))

    data = OrderedDict([(d, []) for d in D])
    best = OrderedDict([(d, None) for d in D])
    guss = OrderedDict([(d, None) for d in D])
    results = list(Pool(ncores).imap_unordered(_bulk_wrapper_core, jobs))
    for (d, c, lc, k, sc, slc, sk) in results:
        data[d].append((c, lc, k, sc, slc, sk))
        if best[d] is None or best[d] > lc:
            best[d] = lc
        if guss[d] is None or guss[d] > slc:
            guss[d] = slc

    print "d,logcost"
    for d in D:
        print "{d:3d},{lc:.1f},{slc:.1f}".format(d=d, lc=best[d], slc=guss[d])

    return data


def whatyougot():
    real = dict()
    idel = dict()
    D = set()
    for fn in os.listdir("probabilities"):
        match = re.match("([0-9]+)_([0-9]+)", fn)
        if not match:
            continue
        d, n = map(int, match.groups())
        if n < 32:
            continue
        D.add(d)
        if not match:
            continue
        if mp.log(n, 2) % 1 != 0:
            continue
        curr =  wrapper(d, n)
        curi =  wrapper(d, n, speculate=True)
        if d not in real or real[d][1] > curr[1]:
            real[d] = curr
        if d not in idel or idel[d][1] > curi[1]:
            idel[d] = curi
    print "d,logcost"
    for d in sorted(D):
        print "{d:3d},{lc:.1f},{slc:.1f}".format(d=d, lc=real[d][1], slc=idel[d][1])
