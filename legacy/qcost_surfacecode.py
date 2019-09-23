# -*- coding: utf-8 -*-
# TODO nothing in this file currently works

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

    lc = wrapper_logical(d, n, k, compute_probs=compute_probs, speculate=speculate)

    # distances, and physical qubits per logical for the layers of distillation
    distances = fifteen_one(p_out=1/mp.mpf(lc.T_count), p_in=mp.mpf(p_in), p_g=mp.mpf(p_g))
    layers = len(distances)
    # NOTE: d_last is used for circuit with biggest logical footprint
    phys_qbits = [num_physical_qubits(distance) for distance in distances[::-1]]
    # physical qubits per layer, starting with topmost
    phys_qbits_layer = [16*(15**(layers-i))*phys_qbits[i-1] for i in range(1, layers + 1)] # noqa

    # total surface code cycles per magic state distillation (not pipelined)
    if layers >= 1:
        scc = 10 * sum(distances)
        total_distil_logi_qbits = 16 * (15 ** (layers - 1))
    else:
        scc = 10
        total_distil_logi_qbits = 1

    # how many magic states can we pipeline at once?
    if layers <= 1:
        parallel_msd = 1
    else:
        parallel_msd = mp.floor(max(float(phys_qbits_layer[0])/phys_qbits_layer[1], 1))

    # number of magic state distilleries required
    msds = mp.ceil(float(lc.T_width)/parallel_msd)

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
    scc_galg = scc * lc.T_depth
    # total cost (ignoring Cliffords in error correction) is
    total_cost_per_galg = total_logi_qbits * scc_galg

    total_cost = lc.total_galgs * total_cost_per_galg

    return float(mp.log(total_cost, 2)), float(mp.log(total_cost/mp.mpf(lc.total_giters), 2)),  k
