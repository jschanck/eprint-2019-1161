#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import re
from mpmath import mp
from optimise_popcnt import giterations_per_grover, load_estimate


def _parse_csv_list_size():
    dim = []
    log_2_list = []

    with open('../scripts/2075.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            dim.append(int(row[0]))
            log_2_list.append(float(row[1]))

    return dim, log_2_list


_, log_2_list = _parse_csv_list_size()


def _preproc_params(d, n, k, compute_probs=True, speculate=False):
    """
    Normalise inputs and check for consistency.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter
    :param compute_probs: compute probabilities if needed
    :param speculate: pretend popcount is perfect

    """

    # For a dimension d lattice using an NV like sieve we expect O(2^{0.2075*d}) lattice vectors.
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

    if d >= 50:
        index_wires = mp.ceil(mp.log(list_growth, 2) + log_2_list[d - 50])
    else:
        index_wires = mp.ceil(mp.log(list_growth * (2**(0.2075*d)), 2))
    if index_wires < 4:
        raise ValueError("diffusion operator poorly defined, d = %d too small."%d)

    # a useful value for computation that follows the writeup
    ell = mp.log(n, 2) + 1

    return d, n, k, index_wires, ell


def popcount_gates(d, k, n):
    d, n, k, _, ell = _preproc_params(d, n, k)
    OR_CNOTs = 2
    OR_Tofs = 1

    # number of ORs required to test whether popcount is less than 2^t, some
    # t in {0, 1, 2, ..., l - 1}, is l - t - 1, i.e. more for smaller
    # t. we say k in [ 2^t + 1 , 2^(t + 1) ] costs the same number of ORs

    t = mp.ceil(mp.log(k, 2))
    ORs = ell - t - 1

    def i_bit_adder_CNOTs(i):
        # to achieve the Toffoli counts for i_bits_adder_Tofs() below we
        # diverge from the expected number of CNOTs for 1 bit adders
        if i == 1:
            return 6
        else:
            return 5*i - 3

    def i_bit_adder_Tofs(i):
        # these Toffoli counts for 1 and 2 bit adders can be achieved using
        # (some of) the optimisations in Cuccarro et al.
        return 2*i - 1

    adder_CNOTs = n*sum([i_bit_adder_CNOTs(i)/float(2**i) for i in range(1, ell)]) # noqa
    popcount_CNOTs = n + OR_CNOTs*ORs + adder_CNOTs

    adder_Tofs = n*sum([i_bit_adder_Tofs(i)/float(2**i) for i in range(1, ell)]) # noqa
    popcount_Tofs = OR_Tofs*ORs + adder_Tofs

    return popcount_CNOTs, popcount_Tofs


def popcount_T_depth(d, k, n):
    d, n, k, index_wires, ell = _preproc_params(d, n, k)

    def i_bit_adder_T_depth(i):
        """
        Each i bit adder has 2i - 1 Toffoli gates (sequentially) so using T
        depth 3 per Toffoli gives 6i - 3 T depth for an i bit adder
        """
        return 6 * i - 3

    # all i bit adders are in parallel and we use 1, ..., log_2(n) bit adders
    adder_T_depth = sum([i_bit_adder_T_depth(i) for i in range(1, ell)])

    # we have ceil(ell - t) OR depth, 1 Tof therefore 3 T-depth each
    t = mp.ceil(mp.log(k, 2))
    OR_T_depth = 3 * mp.ceil(mp.log(ell - t, 2))

    return adder_T_depth,  OR_T_depth


def giteration_T_count(d, n, k):
    """
    T-count for one Grover iteration.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter

    """

    _, _, _, index_wires, _ = _preproc_params(d, n, k)
    _, popcount_Tofs = popcount_gates(d, k, n)

    # each Toffoli costs approx 7T
    popcount_T_count = 7 * popcount_Tofs

    # a l-controlled NOT takes (32l - 84)T
    # the diffusion operator in our circuit is (index_wires - 1)-controlled NOT
    diffusion_T_count = 32 * (index_wires - 1) - 84

    # we have adder and its inverse
    return 2 * popcount_T_count + diffusion_T_count


def giteration_T_depth(d, n, k):
    """
    T-depth of one Grover iteration.

    :param d: sieving dimension
    :param n: number of entries in popcount filter
    :param k: threshold for popcount filter

    """

    d, n, k, index_wires, ell = _preproc_params(d, n, k)

    adder_T_depth, OR_T_depth = popcount_T_depth(d, k, n)

    # I currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    diffusion_T_depth = 32 * (index_wires - 1) - 84

    return 2 * (adder_T_depth + OR_T_depth) + diffusion_T_depth


def giteration_T_width(d, n, k):
    """
    Take the floor (generous to sieving adversary) of the division of T_count
    and T_depth of the given circuit to determine how many T gates required
    on average T depth
    """
    return mp.floor(giteration_T_count(d, n, k)/float(giteration_T_depth(d, n, k)))


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


def clifford_gates(d, n, k, giterations_per_galg):
    """
    Calculate all the Clifford gates required in as many Grover iterations
    as the popcnt parameters require
    """
    d, n, k, index_wires, ell = _preproc_params(d, n, k)

    popcount_CNOTs, _ = popcount_gates(d, k, n)
    diffusion_NOT = 2 * (index_wires - 1)
    diffusion_Hadamard = 2 * (index_wires - 1)
    diffusion_Z = 2

    return (2 * popcount_CNOTs + 1 + diffusion_NOT + diffusion_Hadamard + diffusion_Z) * giterations_per_galg


def distance_condition_clifford(p_in, clifford_gates):
    """
    For the Clifford gates in a Grover iteration, we need a distance as
    calculated below.
    """
    d = 1
    while True:
        if (80 * p_in)**((d + 1)/2.) < 1./clifford_gates:
            break
        d += 1
    return d


def wrapper(d, n, k=None, p_in=10.**(-4), p_g=10.**(-5), compute_probs=True, speculate=False):
    """
    Compute complexity for one sieve iteration using Grover's search.

    :param d: lattice dimension
    :param n: popcount dimension (must be a power of two)
    :param k: threshhold (≈5/16⋅n when ``None``)
    :param p_in:
    :param p_g:
    :param compute_probs: Compute probabilities if they don't exist yet (this can be slow)
    :param speculate: pretend that popcount is optimal

    """
    if k is None:
        best = None
        # NOTE: 5/16*n seems to be optimal
        c = 5 if not speculate else 0
        for k in range(max(int(0.3125*n)-c, 1), min(int(0.3125*n)+c+1, int(n//2))):
            cur = wrapper(d, n, k, p_in=p_in, p_g=p_g, compute_probs=compute_probs, speculate=speculate)
            if best is None or cur < best:
                best = cur
        return best

    _, _, _, index_wires, ell = _preproc_params(d, n, k,
                                                compute_probs=compute_probs,
                                                speculate=speculate)

    # we interpolate between non powers of two n
    assert(mp.log(n, 2)%1 ==0), "Not a power of two n!"

    # calculating the total number of T gates in one full Grover's algorithm, for the required error bound
    if not speculate:
        giterations_per_galg = giterations_per_grover(d, n, k, compute_probs=compute_probs)
    else:
        if d >= 50:
            giterations_per_galg = mp.floor(mp.pi/4*(2**(log_2_list[d - 50]/2.)))
        else:
            giterations_per_galg = mp.floor(mp.pi/4*(2**(0.2075/2*d)))

    galg_T_count = giterations_per_galg * giteration_T_count(d, n, k)
    p_out = mp.mpf('1')/galg_T_count

    p_in = mp.mpf(p_in)
    p_g = mp.mpf(p_g)

    # distances, and physical qubits per logical for the layers of distillation
    distances = fifteen_one(p_out, p_in, p_g=p_g)
    layers = len(distances)
    # NOTE: d_last is used for circuit with biggest logical footprint
    phys_qbits = [num_physical_qubits(distance) for distance in distances[::-1]]

    # physical qubits per layer, starting with topmost
    phys_qbits_layer = [16*(15**(layers-i))*phys_qbits[i-1] for i in range(1, layers + 1)] # noqa

    # total surface code cycles per magic state distillation (not pipelined)
    if len(distances) >= 1:
        scc = 10 * sum(distances)
    else:
        scc = 10

    # total number of physical/logical qubits for msd
    # total_distil_phys_qbits = max(phys_qbits_layer)
    if layers >= 1:
        total_distil_logi_qbits = 16 * (15 ** (layers - 1))
    else:
        total_distil_logi_qbits = 1

    # how many magic states can we pipeline at once?
    if layers <= 1:
        parallel_msd = 1
    else:
        parallel_msd = mp.floor(max(float(phys_qbits_layer[0])/phys_qbits_layer[1], 1))

    # the average T gates per T depth
    T_average = giteration_T_width(d, n, k)

    # number of magic state distilleries required
    msds = mp.ceil(float(T_average)/parallel_msd)

    # logical qubits for Grover's alg = max(width of popcnt circuit, width of diffusion operator) + 1
    t = mp.ceil(mp.log(k, 2))
    logi_qbits_galg = max(3 * n + ell - t - 2, index_wires) + 1

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # distance required for Clifford gates, current ignoring Hadamards in setup
    # galg_clifford_gates = clifford_gates(d, n, k, giterations_per_galg)
    # clifford_distance = distance_condition_clifford(p_in, galg_clifford_gates) # noqa

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # physical qubits for Grover iteration = width * f(clifford_distance)
    # phys_qbits_galg = logi_qbits_galg * num_physical_qubits(clifford_distance) # noqa

    # total number of logical qubits is
    total_logi_qbits = msds * total_distil_logi_qbits + logi_qbits_galg
    # total number of surface code cycles for a grover's algorithm
    scc_galg = giterations_per_galg * scc * giteration_T_depth(d, n, k)
    # total cost (ignoring Cliffords in error correction) is
    total_cost_per_galg = total_logi_qbits * scc_galg

    probs = load_estimate(d, n, k, compute=compute_probs)
    if d >= 50:
        list_size = mp.ceil(2**(log_2_list[d - 50]))
    else:
        list_size = mp.ceil(2**(0.2075*d))
    # c(k, n)
    list_expansion_factor = (probs.ngr_pf/(1 - probs.gr))**(-1./2.)
    # number of galgs to find reduction per vector as popcount not perfect
    repeats = (probs.ngr_pf/probs.pf)**(-1)

    if not speculate:
        total_galgs = list_size * list_expansion_factor * repeats
    else:
        total_galgs = list_size

    total_cost = total_galgs * total_cost_per_galg

    return float(mp.log(total_cost, 2)), float(mp.log(total_cost/(total_galgs * giterations_per_galg), 2)),  k


def _bulk_wrapper_core(args):
    d, n = args
    r = (d,) + wrapper(d, n) + wrapper(d, n, speculate=True)
    print(r)
    return r


def bulk_wrapper(D, N=(32, 64, 128, 256, 512), ncores=1):
    """
    Compute probabilities and establish costs for all pairs ``(d,n) ∈ D × N``.

    :param D: lattice dimensions
    :param N: popcount dimensions
    :param ncores: number of cores to utilise

    """
    from multiprocessing import Pool

    jobs = []
    for n in N:
        for d in D:
            jobs.append((d, n))

    results = list(Pool(ncores).imap_unordered(_bulk_wrapper_core, jobs))
    return results


def overall_estimate(dmod=32):
    """
    Find the best costs for all known probabilities.

    :param dmod: Only consider dimensions ``d`` that are zero mod ``dmod``
    :param nmin: Only consider ``n >= nmin``

    """
    real  = dict()
    ideal = dict()
    D = set()
    for fn in os.listdir("probabilities"):
        match = re.match("([0-9]+)_([0-9]+)", fn)
        if not match:
            continue
        d, n = map(int, match.groups())

        if mp.log(n, 2) % 1 != 0:
            continue
        if d%dmod:
            continue
        try:
            cur_real = wrapper(d, n)
        except IndexError as e:
            print(d, n, e)
            continue
        cur_ideal = wrapper(d, n, speculate=True)
        D.add(d)
        if d not in real or real[d][0] > cur_real[0]:
            real[d] = cur_real
        if d not in ideal or ideal[d][0] > cur_ideal[0]:
            ideal[d] = cur_ideal
    print "d,log cost,log cost per call,log cost opt,log cost per call opt"
    for d in sorted(D):
        print "{d:3d},{lc:.1f},{lcpc:.1f},{slc:.1f},{slcpc:.1f}".format(d=d, lc=real[d][0], slc=ideal[d][0],
                                                                        lcpc=real[d][1], slcpc=ideal[d][1])
