#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpmath import mp
from optimise_popcnt import grover_iterations


def _preproc_params(d, k, n):
    """
    Normalise inputs and check for consistency.

    :param d: sieving dimension
    :param k: threshold for popcount filter
    :param n: number of entries in popcount filter

    """

    # For a dimension d lattice using an NV like sieve we expect Oe(2^{0.2075*d}) lattice vectors.
    # We round this up to the nearest power of two to be able to use Hadamard gates to set up
    # Grover's algorithm.

    # determines width of the diffusion operator
    index_wires = mp.ceil(0.2075*d)
    if index_wires < 4:
        raise ValueError("diffusion operator poorly defined, d = %d too small."%d)

    # TODO: Why is this a problem?
    if index_wires > mp.mpf('3') * n:
        raise ValueError("input (%d) wider than popcnt circuit (%d)!"%(index_wires, 3*n))

    if not 0 <= k <= n:
        raise ValueError("k (%d) not in range 0 ... n (%d)"%(k, n))

    # TODO I doubt we need mpf here.
    d = mp.mpf(d)
    k = mp.mpf(k)
    n = mp.mpf(n)

    return d, k, n, index_wires


def T_count_giteration(d, k, n):
    """
    T-count for one Grover iteration.

    :param d: sieving dimension
    :param k: threshold for popcount filter
    :param n: number of entries in popcount filter

    """

    d, k, n, index_wires = _preproc_params(d, k, n)

    tof_count_adder = n * mp.fraction(7, 2) + mp.log(n, 2) - k
    # each Toffoli costs approx 7T
    T_count_adder = 7 * tof_count_adder

    # a k-controlled NOT (i.e. diffusion operator) in (32k - 84)T
    # TODO: I don't understand this line: index_wires-1 != k
    T_count_diffusion = 32 * (index_wires - 1) - 84

    # we have adder and its inverse
    return 2 * T_count_adder + T_count_diffusion


def T_depth_giteration(d, k, n):
    """
    T-depth of one Grover iteration.

    :param d: sieving dimension
    :param k: threshold for popcount filter
    :param n: number of entries in popcount filter

    """

    d, k, n, index_wires = _preproc_params(d, k, n)

    def T_depth_i_adder(i):
        """
        Each i bit adder has 2i - 1 Toffoli gates (sequentially) so using T
        depth 3 per Toffoli gives 6i - 3 T depth for an i bit adder
        """
        return 6 * i - 3

    # all i bit adders are in parallel and we use 1, ..., log_2 n bit adders
    upper = int(mp.log(n, 2) + 1)
    T_depth_adder = sum([T_depth_i_adder(bits) for bits in range(1, upper)])

    # I currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    # TODO: see ``T_count_giteration`` question
    T_depth_diffusion = 32 * (index_wires - 1) - 84
    return 2 * T_depth_adder + T_depth_diffusion


def T_average_width_giteration(d, k, n):
    """
    Take the floor (generous to sieving adversary) of the division of T_count
    and T_depth of the given circuit to determine how many T gates required
    on average T depth
    """
    # TODO: This isn't taking a floor
    return T_count_giteration(d, k, n)/float(T_depth_giteration(d, k, n))


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


def clifford_gates(d, k, n, total_giterations):
    """
    Calculate all the Clifford gates required in as many Grover iterations
    as the popcnt parameters require
    """
    d, k, n, index_wires = _preproc_params(d, k, n)

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


def wrapper(d, k, n, p_in=10.**(-4), p_g = 10.**(-5)):
    # we will eventually interpolate between non power of two n, sim for k
    # TODO: this doesn't check what it claims it checks
    assert(int(mp.log(n, 2)) % 1 == 0), "Not a power of two n!"
    assert(int(mp.log(k, 2)) % 1 == 0), "Not a power of two k!"

    # calculating the total number of T gates for required error bound
    total_giterations = grover_iterations(d, n, k)
    T_count_total = total_giterations * T_count_giteration(d, k, n)
    p_out = mp.mpf('1')/T_count_total

    p_in = mp.mpf(p_in)
    p_g = mp.mpf(p_g)

    # distances, and physical qubits per logical for the layers of distillation
    distances = fifteen_one(p_out, p_in, p_g=p_g)
    layers = len(distances)
    phys_qbits = [num_physical_qubits(distance) for distance in distances]

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
        parallel_msd = max(float(phys_qbits_layer[0])/phys_qbits_layer[1], 1)

    # the average T gates per T depth
    T_average = T_average_width_giteration(d, k, n)

    # number of magic state distilleries required
    msds = mp.ceil(float(T_average)/parallel_msd)

    # logical qubits for Grover iteration = width of popcnt circuit
    logi_qbits_giteration = 3 * n + mp.log(n, 2) - k - 1

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # distance required for Clifford gates, current ignoring Hadamards in setup
    # giteration_clifford_gates = clifford_gates(d, k, n, total_giterations)
    # clifford_distance = distance_condition_clifford(p_in, giteration_clifford_gates) # noqa

    # NOTE: not used in practice as we don't count surface codes for Cliffords
    # physical qubits for Grover iteration = width * f(clifford_distance)
    # phys_qbits_giteration = (3 * n + mp.log(n, 2) - k - 1) * num_physical_qubits(clifford_distance) # noqa

    # total number of logical qubits is
    total_logi_qbits = msds * total_distil_logi_qbits + logi_qbits_giteration
    # total number of surface code cycles for all the Grover iterations
    total_scc = total_giterations * scc * msds * T_depth_giteration(d, k, n)
    # total cost (ignoring Cliffords in error correction) is
    total_cost = total_logi_qbits * total_scc

    return float(total_cost), float(mp.log(total_cost, 2))
