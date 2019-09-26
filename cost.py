#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum and Classical Nearest Neighbor Cost.
"""

from mpmath import mp
from collections import namedtuple
from utils import load_probabilities, PrecomputationRequired
from config import MagicConstants
from probabilities import W, C, pf
from ge19 import estimate_abstract_to_physical

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
:param toffoli_count: number of Toffoli gates
:param t_count: number of T gates
:param t_depth: T gate depth

"""

LogicalCosts = namedtuple(
    "LogicalCosts",
    (
        "label",
        "qubits_in",
        "qubits_out",
        "qubits_max",  # NOTE : not sure if this is useful
        "depth",
        "gates",
        "dw",
        "toffoli_count",  # NOTE: not sure if this is useful
        "t_count",
        "t_depth",
    ),
)

"""
Classic Costs

:param label: arbitrary label
:param gates: number of gates
:param depth: longest path from input to output

"""

ClassicalCosts = namedtuple("ClassicalCosts", ("label", "gates", "depth"))

"""
METRICS
"""

ClassicalMetrics = {"classical", "naive_classical"}

QuantumMetrics = {
    "g",  # gate count
    "dw",  # depth x width
    "ge19",  # depth x width x physical qubit measurements Gidney Ekera
    "t_count",  # number of T-gates
    "naive_quantum",  # query cost
}

Metrics = ClassicalMetrics | QuantumMetrics


def log2(x):
    return mp.log(x) / mp.log(2)


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
    for k in range(d1, d2 + 1):
        d = 0.1 ** k
        y2 = f(x + d) if x + d < high else f(high)
        if y2 > y:
            d = -d
            y2 = f(x + d) if x + d > low else f(low)
        while (y2 < y) and (low < x + d) and (x + d < high):
            y = y2
            x = x + d
            y2 = f(x)
    return x


def null_costf(qubits_in=0, qubits_out=0):
    """
    Cost of initialization/measurement.
    """

    return LogicalCosts(
        label="null",
        qubits_in=qubits_in,
        qubits_out=qubits_out,
        qubits_max=max(qubits_in, qubits_out),
        gates=0,
        depth=0,
        dw=0,
        toffoli_count=0,
        t_count=0,
        t_depth=0,
    )


def delay(cost, depth, label="_"):
    # delay only affects the dw cost
    dw = cost.dw + cost.qubits_out * depth
    return LogicalCosts(
        label=label,
        qubits_in=cost.qubits_in,
        qubits_out=cost.qubits_out,
        qubits_max=cost.qubits_max,
        gates=cost.gates,
        depth=cost.depth + depth,
        dw=dw,
        toffoli_count=cost.toffoli_count,
        t_count=cost.t_count,
        t_depth=cost.t_depth,
    )


def reverse(cost):
    return LogicalCosts(
        label=cost.label,
        qubits_in=cost.qubits_out,
        qubits_out=cost.qubits_in,
        qubits_max=cost.qubits_max,
        gates=cost.gates,
        depth=cost.depth,
        dw=cost.dw,
        toffoli_count=cost.toffoli_count,
        t_count=cost.t_count,
        t_depth=cost.t_depth,
    )


def compose_k_sequential(cost, times, label="_"):
    # Ensure that sequential composition makes sense
    assert cost.qubits_in == cost.qubits_out

    return LogicalCosts(
        label=label,
        qubits_in=cost.qubits_in,
        qubits_out=cost.qubits_out,
        qubits_max=cost.qubits_max,
        gates=cost.gates * times,
        depth=cost.depth * times,
        dw=cost.dw * times,
        toffoli_count=cost.toffoli_count * times,
        t_count=cost.t_count * times,
        t_depth=cost.t_depth * times,
    )


def compose_k_parallel(cost, times, label="_"):
    return LogicalCosts(
        label=label,
        qubits_in=times * cost.qubits_in,
        qubits_out=times * cost.qubits_out,
        qubits_max=times * cost.qubits_max,
        gates=times * cost.gates,
        depth=cost.depth,
        dw=times * cost.dw,
        toffoli_count=times * cost.toffoli_count,
        t_count=times * cost.t_count,
        t_depth=cost.t_depth,
    )


def compose_sequential(cost1, cost2, label="_"):
    # Ensure that sequential composition makes sense
    assert cost1.qubits_out >= cost2.qubits_in

    # Pad unused wires with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.qubits_out > cost2.qubits_in:
        dw += (cost1.qubits_out - cost2.qubits_in) * cost2.depth
    qubits_out = cost1.qubits_out - cost2.qubits_in + cost2.qubits_out
    qubits_max = max(cost1.qubits_max, cost1.qubits_out - cost2.qubits_in + cost2.qubits_max)

    return LogicalCosts(
        label=label,
        qubits_in=cost1.qubits_in,
        qubits_out=qubits_out,
        qubits_max=qubits_max,
        gates=cost1.gates + cost2.gates,
        depth=cost1.depth + cost2.depth,
        dw=dw,
        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
        t_count=cost1.t_count + cost2.t_count,
        t_depth=cost1.t_depth + cost2.t_depth,
    )


def compose_parallel(cost1, cost2, label="_"):
    # Pad wires from shallower circuit with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.depth >= cost2.depth:
        dw += (cost1.depth - cost2.depth) * cost2.qubits_out
    else:
        dw += (cost2.depth - cost1.depth) * cost1.qubits_out

    return LogicalCosts(
        label=label,
        qubits_in=cost1.qubits_in + cost2.qubits_in,
        qubits_out=cost1.qubits_out + cost2.qubits_out,
        qubits_max=cost1.qubits_max + cost2.qubits_max,
        gates=cost1.gates + cost2.gates,
        depth=max(cost1.depth, cost2.depth),
        dw=dw,
        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
        t_count=cost1.t_count + cost2.t_count,
        t_depth=max(cost1.t_depth, cost2.t_depth),
    )


def classical_popcount_costf(n, k):
    """
    Classical gate count for popcount.

    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    ell = mp.ceil(mp.log(n, 2))
    t = mp.ceil(mp.log(k, 2))
    gates = 11 * n - 9 * ell - 10
    depth = 2 * ell

    cc = ClassicalCosts(label="popcount", gates=gates, depth=depth)

    return cc


def adder_costf(i, ci=False):
    """
    Logical cost of i bit adder (Cuccaro et al). With Carry Input if ci=True

    """
    adder_cnots = 6 if i == 1 else (5 * i + 1 if ci else 5 * i - 3)
    adder_depth = 7 if i == 1 else (2 * i + 6 if ci else 2 * i + 4)
    adder_nots = 0 if i == 1 else (2 * i - 2 if ci else 2 * i - 4)
    adder_tofs = 2 * i - 1
    adder_qubits_in = 2 * i + 1
    adder_qubits_out = 2 * i + 2
    adder_qubits_max = 2 * i + 2
    adder_t_depth = adder_tofs * MagicConstants.t_depth_div_toffoli
    adder_t_count = adder_tofs * MagicConstants.t_div_toffoli
    adder_gates = adder_cnots + adder_nots + adder_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(
        label=str(i) + "-bit adder",
        qubits_in=adder_qubits_in,
        qubits_out=adder_qubits_out,
        qubits_max=adder_qubits_max,
        gates=adder_gates,
        depth=adder_depth,
        dw=adder_qubits_in * adder_depth,
        toffoli_count=adder_tofs,
        t_count=adder_t_count,
        t_depth=adder_t_depth,
    )


def hamming_wt_costf(n):
    """
    Logical cost of mapping |v>|0> to |v>|H(v)>.

    ..  note :: The adder tree uses in-place addition, so some of the bits of |v> overlap |H(v)> and
    there are ancilla as well.

    :param n: number of bits in v

    """
    b = int(mp.floor(log2(n)))
    qc = null_costf(qubits_in=n, qubits_out=n)
    if bin(n + 1).count("1") == 1:
        # When n = 2**(b+1) - 1 the adder tree is "packed". We can use every input bit including
        # carry inputs.
        for i in range(1, b + 1):
            L = compose_k_parallel(adder_costf(i, ci=True), 2 ** (b - i))
            qc = compose_sequential(qc, L)
    else:
        # Decompose into packed adder trees joined by adders.
        # Use one adder tree on (2**b - 1) bits and one on max(1, n - 2**b) bits.
        # Reserve one bit for carry input of adder (unless n = 2**b).
        carry_in = n != 2 ** b
        qc = compose_sequential(
            qc, compose_parallel(hamming_wt_costf(2 ** b - 1), hamming_wt_costf(max(1, n - 2 ** b)))
        )
        qc = compose_sequential(qc, adder_costf(b, ci=carry_in))

    qc = compose_parallel(qc, null_costf(), label=str(n) + "-bit hamming weight")
    return qc


def carry_costf(m):
    """
    Logical cost of mapping |x> to (-1)^{(x+c)_m}|x> where (x+c)_m is the m-th bit (zero indexed) of
    x+c for an arbitrary m bit constant c.

    ..  note :: numbers here are adapted from Fig 3 of https://arxiv.org/pdf/1611.07995.pdf
                m is equivalent to \ell in the LaTeX

    """
    if m < 2:
        raise NotImplementedError("Case m==1 not implemented.")

    carry_cnots = 2 * m
    carry_depth = 8 * m - 8
    carry_nots = 2 * (m - 1)
    carry_tofs = 4 * (m - 2) + 2
    carry_qubits_in = 2 * m
    carry_qubits_out = 2 * m
    carry_qubits_max = 2 * m
    carry_dw = carry_qubits_max * carry_depth
    carry_t_depth = carry_tofs * MagicConstants.t_depth_div_toffoli
    carry_t_count = carry_tofs * MagicConstants.t_div_toffoli
    carry_gates = carry_cnots + carry_nots + carry_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(
        label="carry",
        qubits_in=carry_qubits_in,
        qubits_out=carry_qubits_out,
        qubits_max=carry_qubits_max,
        gates=carry_gates,
        depth=carry_depth,
        dw=carry_dw,
        toffoli_count=carry_tofs,
        t_count=carry_t_count,
        t_depth=carry_t_depth,
    )


def popcount_costf(L, n, k):
    """
    Logical cost of mapping |i> to (-1)^{popcount(u,v_i)}|i> for fixed u.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    assert 0 <= k and k <= n

    index_wires = int(mp.ceil(log2(L)))

    # Initialize space for |v_i>
    qc = null_costf(qubits_in=index_wires, qubits_out=n + index_wires)

    # Query table index i
    # NOTE: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # XOR in the fixed sketch "u"
    # NOTE: We're skipping ~ n NOT gates for mapping |v> to |u^v>
    qc = delay(qc, 1)

    # Use tree of adders compute hamming weight
    #     |i>|u^v_i>|0>     ->    |i>|u^v_i>|wt(u^v_i)>
    hamming_wt = hamming_wt_costf(n)
    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires + hamming_wt.qubits_in))
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

    qc = compose_parallel(qc, null_costf(), label="popcount" + str((n, k)))

    return qc


def n_toffoli_costf(n, have_ancilla=False):
    """
    Logical cost of toffoli with n-1 controls.

    ..  note :: Table I of Maslov arXiv:1508.03273v2 (Source = "Ours", Optimization goal = "T/CNOT")

    """

    assert n >= 3

    if n >= 5 and not have_ancilla:
        # Use Barenco et al (1995) Lemma 7.3 split into two smaller Toffoli gates.
        n1 = int(mp.ceil((n - 1) / 2.0)) + 1
        n2 = n - n1 + 1
        return compose_sequential(
            compose_parallel(null_costf(qubits_in=n - n1, qubits_out=n - n1), n_toffoli_costf(n1, True)),
            compose_parallel(null_costf(qubits_in=n - n2, qubits_out=n - n2), n_toffoli_costf(n2, True)),
        )

    if n == 3:  # Normal toffoli gate
        n_tof_t_count = MagicConstants.AMMR12_tof_t_count
        n_tof_t_depth = MagicConstants.AMMR12_tof_t_depth
        n_tof_gates = MagicConstants.AMMR12_tof_gates
        n_tof_depth = MagicConstants.AMMR12_tof_depth
        n_tof_dw = n_tof_depth * (n + 1)
    elif n == 4: # Note: the cost can be smaller if using "clean" ancillas
                 # (see first "Ours" in Table 1 of Maslov's paper)
        n_tof_t_count = 16
        n_tof_t_depth = 16
        n_tof_gates = 36
        n_tof_depth = 36  # Maslov Eq. (5), Figure 3 (dashed), Eq. (3) (dashed).
        n_tof_dw = n_tof_depth * (n + 1)
    elif n >= 5:
        n_tof_t_count = 8 * n - 16
        n_tof_t_depth = 8 * n - 16
        n_tof_gates = (8 * n - 16) + (8 * n - 20) + (4 * n - 10)
        n_tof_depth = (8 * n - 16) + (8 * n - 20) + (4 * n - 10) 
        n_tof_dw = n_tof_depth * (n + 1)

    n_tof_qubits_max = n if have_ancilla else n + 1

    return LogicalCosts(
        label=str(n) + "-toffoli",
        qubits_in=n,
        qubits_out=n,
        qubits_max=n_tof_qubits_max,
        gates=n_tof_gates,
        depth=n_tof_depth,
        dw=n_tof_dw,
        toffoli_count=0,
        t_count=n_tof_t_count,
        t_depth=n_tof_t_depth,
    )


def diffusion_costf(L):
    """
    Logical cost of the diffusion operator D R_0 D^-1

    where D samples the uniform distribution on {1,...,L} R_0 is the unitary I - 2|0><0|

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    index_wires = int(mp.ceil(log2(L)))

    H = LogicalCosts(
        label="H",
        qubits_in=1,
        qubits_out=1,
        qubits_max=1,
        gates=1,
        depth=1,
        dw=1,
        toffoli_count=0,
        t_count=0,
        t_depth=0,
    )
    Hn = compose_k_parallel(H, index_wires)

    anc = null_costf(qubits_in=index_wires, qubits_out=index_wires + 1)

    qc = compose_sequential(Hn, anc)
    qc = compose_sequential(qc, n_toffoli_costf(index_wires + 1))
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
    :param k: we accept if two vectors agree on <= k

    """

    popcount_cost = popcount_costf(L, n, k)
    diffusion_cost = diffusion_costf(L)

    return compose_sequential(diffusion_cost, popcount_cost, label="oracle")


def popcounts_dominate_cost(positive_rate, d, n, metric):
    ip_div_pc = (MagicConstants.word_size ** 2) * d / float(n)
    if metric in ClassicalMetrics:
        return 1.0 / positive_rate > ip_div_pc
    else:
        return 1.0 / positive_rate > ip_div_pc ** 2


def raw_cost(cost, metric):
    if metric == "g":
        result = cost.gates
    elif metric == "dw":
        result = cost.dw
    elif metric == "ge19":
        phys = estimate_abstract_to_physical(
            cost.toffoli_count, cost.qubits_max, cost.depth, prefers_parallel=False, prefers_serial=True
        )
        result = cost.dw * phys[0] ** 2
    elif metric == "t_count":
        result = cost.t_count
    elif metric == "classical":
        result = cost.gates
    elif metric == "naive_quantum":
        return 1
    elif metric == "naive_classical":
        return 1
    else:
        raise ValueError("Unknown metric '%s'" % metric)
    return result


AllPairsResult = namedtuple("AllPairsResult", ("d", "n", "k", "log_cost", "pf_inv", "metric", "detailed_costs"))


def all_pairs(d, n=None, k=None, optimize=True, metric="dw", allow_suboptimal=False):
    """
    Nearest Neighbor Search via a quadratic search over all pairs.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param epsilon: consider lists of size `(1+ϵ)kC_d(θ)`
    :param optimize: optimize `n`
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters given what is precomputed

    """
    if n is None:
        n = 1
        while n < d:
            n = 2 * n

    k = k if k else int(MagicConstants.k_div_n * (n-1))

    pr = load_probabilities(d, n-1, k)

    def cost(pr):
        N = 2 / ((1 - pr.eta) * C(pr.d, mp.pi / 3))

        if metric in ClassicalMetrics:
            look_cost = classical_popcount_costf(pr.n, pr.k)
            looks = (N ** 2 - N) / 2.0
        else:
            look_cost = popcount_grover_iteration_costf(N, pr.n, pr.k)
            looks_factor = 11.0/15
            looks = int(mp.ceil(looks_factor * N ** (3 / 2.0)))

        full_cost = looks * raw_cost(look_cost, metric)
        return full_cost, look_cost

    positive_rate = pf(pr.d, pr.n, pr.k)
    while optimize and not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
        try:
            pr = load_probabilities(pr.d, 2 * (pr.n+1) - 1, int(MagicConstants.k_div_n * (2 * (pr.n+1) - 1)))
        except PrecomputationRequired as e:
            if allow_suboptimal:
                break
            else:
                raise e
        positive_rate = pf(pr.d, pr.n, pr.k)

    fc, dc = cost(pr)

    return AllPairsResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        log_cost=float(log2(fc)),
        pf_inv=int(round(1 / positive_rate)),
        metric=metric,
        detailed_costs=dc,
    )


RandomBucketsResult = namedtuple(
    "RandomBucketsResult", ("d", "n", "k", "theta", "log_cost", "pf_inv", "metric", "detailed_costs")
)


def random_buckets(d, n=None, k=None, theta1=None, optimize=True, metric="dw", allow_suboptimal=False):
    """
    Nearest Neighbor Search using random buckets as in BGJ1.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta1: bucket angle
    :param optimize: optimize `n`
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters
        given what is precomputed

    """
    if n is None:
        n = 1
        while n < d:
            n = 2 * n

    k = k if k else int(MagicConstants.k_div_n * (n-1))
    theta = theta1 if theta1 else 1.2860
    pr = load_probabilities(d, n-1, k)
    ip_cost = MagicConstants.word_size ** 2 * d

    def cost(pr, T1):
        N = 2 / ((1 - pr.eta) * C(pr.d, mp.pi / 3))
        W0 = W(pr.d, T1, T1, mp.pi / 3)
        buckets = 1.0 / W0
        bucket_size = N * C(pr.d, T1)

        if metric in ClassicalMetrics:
            look_cost = classical_popcount_costf(pr.n, pr.k)
            looks_per_bucket = (bucket_size ** 2 - bucket_size) / 2.0
        else:
            look_cost = popcount_grover_iteration_costf(N, pr.n, pr.k)
            looks_factor = (2 * W0) / (5 * C(pr.d, T1)) + 1.0 / 3
            looks_per_bucket = looks_factor * bucket_size ** (3 / 2.0)

        fill_bucket_cost = N * ip_cost
        search_bucket_cost = looks_per_bucket * raw_cost(look_cost, metric)
        full_cost = buckets * (fill_bucket_cost + search_bucket_cost)

        return full_cost, look_cost

    if optimize:
        theta = local_min(lambda T: cost(pr, T)[0], theta, low=mp.pi / 6, high=mp.pi / 2)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
            try:
                pr = load_probabilities(pr.d, 2 * (pr.n + 1) - 1, int(MagicConstants.k_div_n * (2 * (pr.n + 1) - 1)))
            except PrecomputationRequired as e:
                if allow_suboptimal:
                    break
                else:
                    raise e
            theta = local_min(lambda T: cost(pr, T)[0], theta, low=mp.pi / 6, high=mp.pi / 2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    fc, dc = cost(pr, theta)

    return RandomBucketsResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        theta=float(theta),
        log_cost=float(log2(fc)),
        pf_inv=int(round(1 / positive_rate)),
        metric=metric,
        detailed_costs=dc,
    )


ListDecodingResult = namedtuple(
    "ListDecodingResult", ("d", "n", "k", "theta1", "theta2", "log_cost", "pf_inv", "metric", "detailed_costs")
)


def list_decoding(d, n=None, k=None, theta1=None, theta2=None, optimize=True, metric="dw", allow_suboptimal=False):
    """
    Nearest Neighbor Search via a decodable buckets as in BDGL16.

    :param d: search in \(S^{d-1}\)
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta1: filter creation angle
    :param theta2: filter query angle
    :param optimize: optimize `n`
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters
        given what is precomputed
    """

    if n is None:
        n = 1
        while n < d:
            n = 2 * n

    k = k if k else int(MagicConstants.k_div_n * (n-1))
    theta = theta1 if theta1 else mp.pi / 3
    pr = load_probabilities(d, n-1, k)
    ip_cost = MagicConstants.word_size ** 2 * d

    def cost(pr, T1):
        T2 = T1
        N = 2 / ((1 - pr.eta) * C(d, mp.pi / 3))
        W0 = W(d, T1, T2, mp.pi / 3)
        filters = 1.0 / W0
        insert_cost = filters * C(d, T2) * ip_cost
        query_cost = filters * C(d, T1) * ip_cost
        bucket_size = (filters * C(d, T1)) * (N * C(d, T2))

        if metric in ClassicalMetrics:
            look_cost = classical_popcount_costf(pr.n, pr.k)
            looks_per_bucket = bucket_size
            search_one_cost = ClassicalCosts(
                label="search", gates=look_cost.gates * looks_per_bucket, depth=look_cost.depth * looks_per_bucket
            )
        else:
            look_cost = popcount_grover_iteration_costf(N, pr.n, pr.k)
            looks_per_bucket = bucket_size ** (1 / 2.0)
            search_one_cost = compose_k_sequential(look_cost, looks_per_bucket)

        search_cost = raw_cost(search_one_cost, metric)
        return N * insert_cost + N * query_cost + N * search_cost, search_one_cost

    if optimize:
        theta = local_min(lambda T: cost(pr, T)[0], theta, low=mp.pi / 6, high=mp.pi / 2)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
            try:
                pr = load_probabilities(pr.d, 2 * (pr.n + 1) - 1, int(MagicConstants.k_div_n * (2 * (pr.n + 1) - 1)))
            except PrecomputationRequired as e:
                if allow_suboptimal:
                    break
                else:
                    raise e
            theta = local_min(lambda T: cost(pr, T)[0], theta, low=mp.pi / 6, high=mp.pi / 2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    fc, dc = cost(pr, theta)

    return ListDecodingResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        theta1=float(theta),
        theta2=float(theta),
        log_cost=float(log2(fc)),
        pf_inv=int(round(1 / positive_rate)),
        metric=metric,
        detailed_costs=dc,
    )
