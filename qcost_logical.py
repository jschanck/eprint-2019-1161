#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum Sieving Cost on Logical Layer.
"""

from mpmath import mp
from collections import namedtuple
from utils import load_probabilities
from config import MagicConstants
from probabilities_estimates import W, C, pf

"""
COSTS
"""

"""
Logical Quantum Costs

:param label: arbitrary label
:param qubits_in: number of input qubits
:param qubits_out: number of output qubits
:param qubits_max:
:param depth: longest path from input to output (including identity gates)
:param gates: gates except identity gates
:param dw: not necessarily depth*qubits
:param toffoli_count:
:param t_count:
:param t_depth:

"""

LogicalCosts = namedtuple("LogicalCosts",
                          ("label",
                           "qubits_in",
                           "qubits_out",
                           "qubits_max",  # NOTE : not sure if this is useful
                           "depth",
                           "gates",
                           "dw",
                           "toffoli_count",  # NOTE: not sure if this is useful
                           "t_count",
                           "t_depth"))

# NOTE: unused
# PhysicalCosts = namedtuple("PhysicalCosts", ("label", "physical_qubits", "surface_code_cycles")) # What else?

"""
Classic Costs

:param label: arbitrary label
:param gates: number of gates
:param depth: longest path from input to output

"""

ClassicalCosts = namedtuple("ClassicalCosts",
                            ("label",
                             "gates",
                             "depth"))

"""
METRICS
"""

ClassicalMetrics = {"classical",
                    "naive_classical"}

QuantumMetrics = {"g",   # gate count
                  "dw",  # depth × width
                  "t_count",  # number of T-gates
                  "naive_quantum"  # TODO: document
                  }

Metrics = ClassicalMetrics | QuantumMetrics


def log2(x):
    return mp.log(x)/mp.log(2)


def local_min(f, x, d1=1, d2=5, low=None, high=None):
    """
    Search the neighborhood around ``f(x)`` for a local minimum between ``low`` and ``high``.

    ..  note :: We could replace this function with a call to ``scipy.optimize.fminbound``.
    However, this doesn't seem to be faster.  Also, ``f(x)`` is not necessarily defined on all of
    `[low … high]`, raising an ``AssertionError`` which would need to be caught by a wrapper around
    ``f``.

    :param f: function to call
    :param x: initial guess for ``x`` minimizing ``f(x)``
    :param d1: We move in steps of size `0.1^{d1}` … 0.1^{d2}`, starting with ``d1``
    :param d2: We move in steps of size `0.1^{d1}` … 0.1^{d2}`, finishing at ``d2``
    :param low: lower bound on input space
    :param high: upper bound on input space

    """
    y = f(x)
    for k in range(d1, d2+1):
        d = 0.1**k
        y2 = f(x+d) if x+d < high else f(high)
        if y2 > y:
            d = -d
            y2 = f(x+d) if x+d > low else f(low)
        while y2 < y and low < x+d and x+d < high:
            y = y2
            x = x+d
            y2 = f(x+d)
    return x


def null_costf(qubits_in=0, qubits_out=0):
    """
    Cost of initialization/measurement.
    """
    # TODO: Should we charge gates / depth for those?

    return LogicalCosts(label="null",
                        qubits_in=qubits_in,
                        qubits_out=qubits_out,
                        qubits_max=max(qubits_in, qubits_out),
                        gates=0,
                        depth=0,
                        dw=0,
                        toffoli_count=0,
                        t_count=0,
                        t_depth=0)


def delay(cost, depth, label="_"):
    # delay only affects the dw cost
    dw = cost.dw + cost.qubits_out*depth
    return LogicalCosts(label=label,
                        qubits_in=cost.qubits_in,
                        qubits_out=cost.qubits_out,
                        qubits_max=cost.qubits_max,
                        gates=cost.gates,
                        depth=cost.depth+depth,
                        dw=dw,
                        toffoli_count=cost.toffoli_count,
                        t_count=cost.t_count,
                        t_depth=cost.t_depth)


def reverse(cost):
    return LogicalCosts(label=cost.label,
                        qubits_in=cost.qubits_out,
                        qubits_out=cost.qubits_in,
                        qubits_max=cost.qubits_max,
                        gates=cost.gates,
                        depth=cost.depth,
                        dw=cost.dw,
                        toffoli_count=cost.toffoli_count,
                        t_count=cost.t_count,
                        t_depth=cost.t_depth)


def compose_k_sequential(cost, times, label="_"):
    # Ensure that sequential composition makes sense
    assert cost.qubits_in == cost.qubits_out

    return LogicalCosts(label=label,
                        qubits_in=cost.qubits_in,
                        qubits_out=cost.qubits_out,
                        qubits_max=cost.qubits_max,
                        gates=cost.gates*times,
                        depth=cost.depth*times,
                        dw=cost.dw*times,
                        toffoli_count=cost.toffoli_count*times,
                        t_count=cost.t_count*times,
                        t_depth=cost.t_depth*times)


def compose_k_parallel(cost, times, label="_"):
    return LogicalCosts(label=label,
                        qubits_in=times * cost.qubits_in,
                        qubits_out=times * cost.qubits_out,
                        qubits_max=times * cost.qubits_max,
                        gates=times * cost.gates,
                        depth=cost.depth,
                        dw=times * cost.dw,
                        toffoli_count=times * cost.toffoli_count,
                        t_count=times * cost.t_count,
                        t_depth=cost.t_depth)


def compose_sequential(cost1, cost2, label="_"):
    # Ensure that sequential composition makes sense
    assert cost1.qubits_out >= cost2.qubits_in

    # Pad unused wires with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.qubits_out > cost2.qubits_in:
        dw += (cost1.qubits_out - cost2.qubits_in) * cost2.depth
    qubits_out = cost1.qubits_out - cost2.qubits_in + cost2.qubits_out
    qubits_max = max(cost1.qubits_max, cost1.qubits_out - cost2.qubits_in + cost2.qubits_max)

    return LogicalCosts(label=label,
                        qubits_in=cost1.qubits_in,
                        qubits_out=qubits_out,
                        qubits_max=qubits_max,
                        gates=cost1.gates + cost2.gates,
                        depth=cost1.depth + cost2.depth,
                        dw=dw,
                        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
                        t_count=cost1.t_count + cost2.t_count,
                        t_depth=cost1.t_depth + cost2.t_depth)


def compose_parallel(cost1, cost2, label="_"):
    # Pad wires from shallower circuit with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.depth >= cost2.depth:
        dw += (cost1.depth - cost2.depth) * cost2.qubits_out
    else:
        dw += (cost2.depth - cost1.depth) * cost1.qubits_out

    return LogicalCosts(label=label,
                        qubits_in=cost1.qubits_in + cost2.qubits_in,
                        qubits_out=cost1.qubits_out + cost2.qubits_out,
                        qubits_max=cost1.qubits_max + cost2.qubits_max,
                        gates=cost1.gates + cost2.gates,
                        depth=max(cost1.depth, cost2.depth),
                        dw=dw,
                        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
                        t_count=cost1.t_count + cost2.t_count,
                        t_depth=max(cost1.t_depth, cost2.t_depth))


def classical_popcount_costf(n, k):
    ell = mp.ceil(mp.log(n, 2)+1)
    t = mp.ceil(mp.log(k, 2))
    gates = 10*n - 9*ell - t - 2
    depth = 1 + 2*ell + 2 + mp.ceil(mp.log(ell - t - 1, 2))

    cc = ClassicalCosts(label="popcount",
                        gates=gates,
                        depth=depth)

    return cc


def adder_costf(i, ci=False):
    """
    Logical cost of i bit adder (Cuccaro et al). With Carry Input if ci=True

    """
    adder_cnots = 6 if i == 1 else (5*i+1 if ci else 5*i-3)
    adder_depth = 7 if i == 1 else (2*i+6 if ci else 2*i+4)
    adder_nots  = 0 if i == 1 else (2*i-2 if ci else 2*i-4)
    adder_tofs  = 2*i-1
    adder_qubits_in = 2*i+1 if ci else 2*i
    adder_qubits_out = 2*i+2 if ci else 2*i+1
    adder_qubits_max = 2*i+2
    adder_t_depth = adder_tofs * MagicConstants.t_depth_div_toffoli
    adder_t_count = adder_tofs * MagicConstants.t_div_toffoli
    adder_gates = adder_cnots + adder_nots + adder_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(label=str(i)+"-bit adder",
                        qubits_in=adder_qubits_in,
                        qubits_out=adder_qubits_out,
                        qubits_max=adder_qubits_max,
                        gates=adder_gates,
                        depth=adder_depth,
                        dw=adder_qubits_in * adder_depth,
                        toffoli_count=adder_tofs,
                        t_count=adder_t_count,
                        t_depth=adder_t_depth)


def hamming_wt_costf(n):
    """
    Logical cost of mapping |v>|0> to |v>|H(v)>.

    ..  note :: The adder tree uses in-place addition, so some of the bits of |v> overlap |H(v)> and
    there are ancilla as well.

    :param n: number of bits in v

    """
    b = int(mp.floor(log2(n)))
    qc = null_costf(qubits_in=n, qubits_out=n)
    if bin(n+1).count('1') == 1:
        # When n = 2**(b+1) - 1 the adder tree is "packed". We can use every input bit including
        # carry inputs.
        for i in range(1, b + 1):
            L = compose_k_parallel(adder_costf(i, ci=True), 2**(b-i))
            qc = compose_sequential(qc, L)
    else:
        # Decompose into packed adder trees joined by adders.
        # Use one adder tree on (2**b - 1) bits and one on max(1, n - 2**b) bits.
        # Reserve one bit for carry input of adder (unless n = 2**b).
        carry_in = (n != 2**b)
        qc = compose_sequential(qc,
                                compose_parallel(hamming_wt_costf(2**b-1),
                                                 hamming_wt_costf(max(1, n-2**b))))
        qc = compose_sequential(qc, adder_costf(b, ci=carry_in))

    qc = compose_parallel(qc, null_costf(), label=str(n)+"-bit hamming weight")
    return qc


def carry_costf(m):
    """
    Logical cost of mapping |x> to (-1)^{(x+c)_m}|x> where (x+c)_m is the m-th bit (zero indexed) of
    x+c for an arbitrary m bit constant c.

    ..  note :: Numbers here are for "high bit only" circuit from Cuccaro et al
    """
    if m < 2:
        raise NotImplementedError("Case m==1 not implemented.")

    carry_cnots = 4*m-3
    carry_depth = 2*m+3
    carry_nots  = 0
    carry_tofs  = 2*m-1
    carry_qubits_in = m
    carry_qubits_out = m
    carry_qubits_max = 2*m+1
    carry_dw = carry_qubits_max * carry_depth
    carry_t_depth = carry_tofs * MagicConstants.t_depth_div_toffoli
    carry_t_count = carry_tofs * MagicConstants.t_div_toffoli
    carry_gates = carry_cnots + carry_nots + carry_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(label="carry",
                        qubits_in=carry_qubits_in,
                        qubits_out=carry_qubits_out,
                        qubits_max=carry_qubits_max,
                        gates=carry_gates,
                        depth=carry_depth,
                        dw=carry_dw,
                        toffoli_count=carry_tofs,
                        t_count=carry_t_count,
                        t_depth=carry_t_depth)


def popcount_costf(L, n, k):
    """
    Logical cost of mapping |i> to (-1)^{popcount(u,v_i)}|i> for fixed u.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    assert 0 <= k and k <= n

    index_wires = int(mp.ceil(log2(L)))

    # Initialize space for |v_i>
    qc = null_costf(qubits_in=index_wires, qubits_out=n+index_wires)

    # Query table index i
    # NOTE: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # XOR in the fixed sketch "u"
    # NOTE: We're skipping ~ n NOT gates for mapping |v> to |u^v>
    qc = delay(qc, 1)

    # Use tree of adders compute hamming weight
    #     |i>|u^v_i>|0>     ->    |i>|u^v_i>|wt(u^v_i)>
    hamming_wt = hamming_wt_costf(n)
    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires+hamming_wt.qubits_in))
    qc = compose_sequential(qc, hamming_wt)

    # Compute the high bit of (2^ceil(log(n)) - k) + hamming_wt
    #     |i>|v_i>|wt(u^v_i)>   ->     (-1)^popcnt(u,v_i) |i>|u^v_i>|wt(u^v_i)>
    qc = compose_sequential(qc, carry_costf(int(mp.ceil(log2(n)))))

    # Uncompute hamming weight.
    qc = compose_sequential(qc, reverse(hamming_wt))

    # Uncompute XOR
    # NOTE: We're skipping ~ n NOT gates for mapping |u^v> to |v>
    qc = delay(qc, 1)

    # Uncompute table entry
    # NOTE: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # Discard ancilla
    # (-1)^popcnt(u,v_i) |i>|0>|0>   ->    (-1)^popcnt(u,v_i) |i>

    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires))

    qc = compose_parallel(qc, null_costf(), label="popcount"+str((n, k)))

    return qc


def n_toffoli_costf(n, have_ancilla=False):
    """
    Logical cost of toffoli with n-1 controls.

    ..  note :: Table I of Maslov arXiv:1508.03273v2 (Source = "Ours", Optimization goal = "T/CNOT")
    """

    # TODO: This needs to be reviewed.

    assert n >= 3

    if n >= 5 and not have_ancilla:
        # Use Barenco et al (1995) Lemma 7.3 split into two smaller Toffoli gates.
        n1 = int(mp.ceil((n-1)/2.0)) + 1
        n2 = n - n1 + 1
        return compose_sequential(
            compose_parallel(null_costf(qubits_in=n-n1, qubits_out=n-n1), n_toffoli_costf(n1, True)),
            compose_parallel(null_costf(qubits_in=n-n2, qubits_out=n-n2), n_toffoli_costf(n2, True)))

    if n == 3:  # Normal toffoli gate
        n_tof_t_count = MagicConstants.AMMR12_tof_t_count
        n_tof_t_depth = MagicConstants.AMMR12_tof_t_depth
        n_tof_gates   = MagicConstants.AMMR12_tof_gates
        n_tof_depth   = MagicConstants.AMMR12_tof_depth
        n_tof_dw      = n_tof_depth * (n+1)
    elif n == 4:
        n_tof_t_count = 16
        n_tof_t_depth = 16
        n_tof_gates   = 36
        n_tof_depth   = 36  # Maslov Eq. (5), Figure 3 (dashed), Eq. (3) (dashed).
        n_tof_dw      = n_tof_depth * (n+1)
    elif n >= 5:
        n_tof_t_count = 8*n-16
        n_tof_t_depth = 8*n-16
        n_tof_gates   = (8*n-16) + (8*n-20) + (4*n-10)
        n_tof_depth   = (8*n-16) + (8*n-20) + (4*n-10)  # TODO: check
        n_tof_dw      = n_tof_depth * (n+1)

    n_tof_qubits_max = n if have_ancilla else n+1

    return LogicalCosts(label=str(n)+"-toffoli",
                        qubits_in=n,
                        qubits_out=n,
                        qubits_max=n_tof_qubits_max,
                        gates=n_tof_gates,
                        depth=n_tof_depth,
                        dw=n_tof_dw,
                        toffoli_count=0,
                        t_count=n_tof_t_count,
                        t_depth=n_tof_t_depth)


def diffusion_costf(L):
    """
    Logical cost of the diffusion operator D R_0 D^-1

    where D samples the uniform distribution on {1,...,L} R_0 is the unitary I - 2|0><0|

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    index_wires = int(mp.ceil(log2(L)))

    H = LogicalCosts(label="H", qubits_in=1, qubits_out=1, qubits_max=1,
                     gates=1, depth=1, dw=1,
                     toffoli_count=0, t_count=0, t_depth=0)
    Hn = compose_k_parallel(H, index_wires)

    anc = null_costf(qubits_in=index_wires, qubits_out=index_wires+1)

    qc = compose_sequential(Hn, anc)
    qc = compose_sequential(qc, n_toffoli_costf(index_wires+1))
    qc = compose_sequential(qc, reverse(anc))
    qc = compose_sequential(qc, Hn)

    qc = compose_parallel(qc, null_costf(), label="diffusion")
    return qc


def popcount_grover_iteration_costf(L, n, k):
    """
    Logical cost of G(popcount) = (D R_0 D^-1) R_popcount.

    where D samples the uniform distribution on {1,...,L} (D R_0 D^-1) is the diffusion operator.
    R_popcount maps |i> to (-1)^{popcount(u,v_i)}|i> for some fixed u

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """

    popcount_cost = popcount_costf(L, n, k)
    diffusion_cost = diffusion_costf(L)

    return compose_sequential(diffusion_cost, popcount_cost, label="oracle")


def searchf(L, n, k):
    """
    Logical cost of popcount filtered search that succeeds with high probability.

    The search routine takes two integer parameters m1 and m2.
    We pick a random number i in {0, ..., m1-1} and another j in {0, ..., m2-1}.
    We then apply the unitary
      G(G(pc)^i D, ip)^j =
        ((G(pc)^i D) R_0 (G(pc)^i D)^-1 R_ip)^j G(pc)^i D
    (Although here we only cost G(pc)^ij and G(pc)^-ij.)

    Let P be the number of popcount positives.

    The routine uses an expected m1/2 iterations of G(pc) for the sampling
    routine in amplitude amplification. AA calls the sampling routine twice, so we
    use an expected m1 popcount oracles per AA iteration.
    Since we don't know P exactly, we're going to leave some probability mass
    on popcount negatives. We have to account for that in the AA step.

    Assume m1 > pi/4 * sqrt(L/P). Then the probability mass assigned to popcount
    positives after G(pc)^i is at least 1/filter_amplification_factor^2. Suppose
    m2 > pi/4 * filter_amplification_factor * sqrt(P).
    Then we expect to succeed with probability (at least)
        1/2 * 1/filter_repetition_factor
    after an expected
        m2 / 2
    AA iterations.

    So we succeed with probability ~ 1 after 2*filter_repetition_factor repetitions.

    We assume the best case for the adversary,
        m1=pi/4*sqrt(L/P) and m2=pi/4*filter_amplification_factor*sqrt(P).
    The total number of popcount oracle calls is
        (pi/4)^2 * filter_amplification_factor * filter_repetition_factor * sqrt(L).
    This is about 2 * sqrt(L).

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    qc = popcount_grover_iteration_costf(L, n, k)

    count  = mp.sqrt(L)
    count *= (mp.pi/4)*(mp.pi/4)
    count *= MagicConstants.filter_amplification_factor
    count *= MagicConstants.filter_repetition_factor

    return compose_k_sequential(qc, count, label="search")


def popcounts_dominate_cost(positive_rate, metric):
    if metric in ClassicalMetrics:
        return 1.0/positive_rate > MagicConstants.ip_div_pc
    else:
        # TODO: Double check that this is what we want. Quantum search does sqrt(1/positive_rate)
        # popcounts per inner product.
        return 1.0/positive_rate > MagicConstants.ip_div_pc**2


AllPairsResult = namedtuple("AllPairsResult", ("d", "n", "k", "log_cost", "pf_inv", "metric"))


def all_pairs(d, n=None, k=None, epsilon=0.01, optimize=True, metric="dw"):
    """
    Nearest Neighbor Search via a quadratic search over all pairs.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param epsilon: consider lists of size `(1+ϵ)kC_d(θ)`
    :param optimize: optimize `n`
    :param metric: target metric

    """
    if n is None:
        n = 1
        while n < d:
            n = 2*n
    k = k if k else int(MagicConstants.k_div_n * n)

    pr = load_probabilities(d, n, k)

    epsilon = mp.mpf(epsilon)

    def cost(pr):
        L = (1+epsilon) * 2/((1-pr.eta)*C(pr.d, mp.pi/3))
        search_calls = int(mp.ceil(2*(1+epsilon)/(1-pr.eta) * L))
        expected_bucket_size = ((1-pr.eta)/(1+epsilon))**2 * L/2
        if metric == "g":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).gates
        elif metric == "dw":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).dw
        elif metric == "t_count":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).t_count
        elif metric == "naive_quantum":
            search_cost = mp.sqrt(expected_bucket_size)
        elif metric == "classical":
            search_cost = expected_bucket_size * classical_popcount_costf(pr.n, pr.k).gates
        elif metric == "naive_classical":
            search_cost = expected_bucket_size
        else:
            raise ValueError("Unknown metric '%s'"%metric)
        return search_calls * search_cost

    positive_rate = pf(pr.d, pr.n, pr.k)
    while optimize and not popcounts_dominate_cost(positive_rate, metric):
        pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
        positive_rate = pf(pr.d, pr.n, pr.k)

    return AllPairsResult(d=pr.d,
                          n=pr.n,
                          k=pr.k,
                          log_cost=float(log2(cost(pr))),
                          pf_inv=int(round(1/positive_rate)),
                          metric=metric)


RandomBucketsResult = namedtuple("RandomBucketsResult", ("d", "n", "k", "theta", "log_cost", "pf_inv", "metric"))


def random_buckets(d, n=None, k=None, theta1=None, optimize=True, metric="dw"):
    """
    Nearest Neighbor Search using random buckets as in BGJ1.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta1: bucket angle
    :param optimize: optimize `n`
    :param metric: target metric

    """
    if n is None:
        n = 1
        while n < d:
            n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    theta = theta1 if theta1 else 1.2860
    pr = load_probabilities(d, n, k)
    # NOTE: We assume checking whether a vector belongs in a bucket costs a popcount call
    fill_cost_per_call = classical_popcount_costf(pr.n, pr.k).gates

    def cost(pr, T1):
        L = 2/((1-pr.eta)*C(pr.d, mp.pi/3))
        buckets = 1.0/W(pr.d, T1, T1, mp.pi/3)
        expected_bucket_size = L * C(pr.d, T1)
        fill_cost = L * fill_cost_per_call
        average_search_size = expected_bucket_size/2
        searches_per_bucket = expected_bucket_size
        if metric == "g":
            search_cost = searchf(average_search_size, pr.n, pr.k).gates
        elif metric == "dw":
            search_cost = searchf(average_search_size, pr.n, pr.k).dw
        elif metric == "t_count":
            search_cost = searchf(average_search_size, pr.n, pr.k).t_count
        elif metric == "naive_quantum":
            search_cost = mp.sqrt(average_search_size)
        elif metric == "classical":
            search_cost = average_search_size * classical_popcount_costf(pr.n, pr.k).gates
        elif metric == "naive_classical":
            search_cost = average_search_size
        else:
            raise ValueError("Unknown metric")
        return buckets * (searches_per_bucket * search_cost + fill_cost)

    if optimize:
        theta = local_min(lambda T: cost(pr, T), theta, low=mp.pi/6, high=mp.pi/2)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, metric):
            pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
            theta = local_min(lambda T: cost(pr, T), theta, low=mp.pi/6, high=mp.pi/2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return RandomBucketsResult(d=pr.d,
                               n=pr.n,
                               k=pr.k,
                               theta=float(theta),
                               log_cost=float(log2(cost(pr, theta))),
                               pf_inv=int(round(1/positive_rate)),
                               metric=metric)


TableBucketsResult = namedtuple("TableBucketsResult", ("d", "n", "k", "theta1", "theta2", "log_cost", "pf_inv", "metric"))


def table_buckets(d, n=None, k=None, theta1=None, theta2=None, optimize=True, metric="dw"):
    """
    Nearest Neighbor Search via a decodable buckets as in BDGL16.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta1: filter creation angle
    :param theta2: filter query angle
    :param optimize: optimize `n`
    :param metric: target metric

    """

    if n is None:
        n = 1
        while n < d:
            n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    theta = theta1 if theta1 else mp.pi/3
    pr = load_probabilities(d, n, k)
    # NOTE: We assume checking whether a vector belongs in a bucket costs a popcount call
    fill_cost_per_call = classical_popcount_costf(pr.n, pr.k).gates

    def cost(pr, T1):
        T2 = T1  # TODO: Handle theta1 != theta2
        L = 2/((1-pr.eta)*C(d, mp.pi/3))
        search_calls = int(mp.ceil(L))
        filters = 1/W(d, T1, T2, mp.pi/3)
        populate_table_cost = L * filters * C(d, T2) * fill_cost_per_call
        relevant_bucket_cost = filters * C(d, T1) * fill_cost_per_call
        average_search_size = L * filters * C(d, T1) * C(d, T2) / 2
        # TODO: Scale insert_cost and relevant_bucket_cost?
        if metric == "g":
            search_cost = searchf(average_search_size, pr.n, pr.k).gates
        elif metric == "dw":
            search_cost = searchf(average_search_size, pr.n, pr.k).dw
        elif metric == "t_count":
            search_cost = searchf(average_search_size, pr.n, pr.k).t_count
        elif metric == "naive_quantum":
            search_cost = mp.sqrt(average_search_size)
        elif metric == "classical":
            search_cost = average_search_size * classical_popcount_costf(pr.n, pr.k).gates
        elif metric == "naive_classical":
            search_cost = average_search_size
        else:
            raise ValueError("Unknown metric")
        return search_calls * (search_cost + relevant_bucket_cost) + populate_table_cost

    if optimize:
        theta = local_min(lambda T: cost(pr, T), theta, low=mp.pi/6, high=mp.pi/2)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, metric):
            pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
            theta = local_min(lambda T: cost(pr, T), theta, low=mp.pi/6, high=mp.pi/2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return TableBucketsResult(d=pr.d,
                              n=pr.n,
                              k=pr.k,
                              theta1=float(theta),
                              theta2=float(theta),
                              log_cost=float(log2(cost(pr, theta))),
                              pf_inv=int(round(1/positive_rate)),
                              metric=metric)
