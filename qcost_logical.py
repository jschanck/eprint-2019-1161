#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mpmath import mp
from collections import namedtuple
from utils import load_probabilities
from config import MagicConstants
from probabilities_estimates import W, C, pf


# NOTE: "qubits_in" is the number of input qubits.
#       "qubits_out" is the number of output qubits.
#       "gates" does not include identity gates
#       "depth" is longest path from input to output (including identity gates)
#       "dw" is not necessarily depth*qubits
# XXX:  "toffoli_count" is not very useful. Remove it?
LogicalCosts = namedtuple("LogicalCosts",
                          ("label",
                           "qubits_in", "qubits_out",
                           "depth", "gates", "dw",
                           "toffoli_count",
                           "t_count", "t_depth"))

PhysicalCosts = namedtuple("PhysicalCosts", ("label", "physical_qubits", "surface_code_cycles")) # What else?

ClassicalCosts = namedtuple("ClassicalCosts", ("label", "gates", "depth")) # What else?

ClassicalMetrics = {"classical", "naive_classical"}
QuantumMetrics = {"G", "DW", "t_count", "naive_quantum"}
Metrics = ClassicalMetrics | QuantumMetrics


def log2(x):
    return mp.log(x)/mp.log(2)


def local_min(f,x,D1=1,D2=5,LOW=None, HIGH=None):
    y = f(x)
    for k in range(D1, D2+1):
        d = 0.1**k
        y2 = f(x+d) if x+d < HIGH else f(HIGH)
        if y2 > y:
            d = -d
            y2 = f(x+d) if x+d > LOW else f(LOW)
        while y2 < y and LOW < x+d and x+d < HIGH:
            y = y2
            x = x+d
            y2 = f(x+d)
    return x


def null_costf(qubits_in=0, qubits_out=0):
    # XXX: This gets used for initialisation/measurement. Should we charge gates / depth for those?
    return LogicalCosts(label="null",
                        qubits_in=qubits_in, qubits_out=qubits_out,
                        gates=0, depth=0, dw=0,
                        toffoli_count=0, t_count=0, t_depth=0)

def delay(cost, depth, label="_"):
    return LogicalCosts(label=label,
                        qubits_in=cost.qubits_in,
                        qubits_out=cost.qubits_out,
                        gates=cost.gates,
                        depth=cost.depth+depth,
                        dw=cost.dw + cost.qubits_out*depth,
                        toffoli_count=cost.toffoli_count,
                        t_count=cost.t_count,
                        t_depth=cost.t_depth)

def reverse(cost):
    return LogicalCosts(label=cost.label,
                        qubits_in=cost.qubits_out,
                        qubits_out=cost.qubits_in,
                        gates=cost.gates,
                        depth=cost.depth,
                        dw=cost.dw,
                        toffoli_count=cost.toffoli_count,
                        t_count=cost.t_count,
                        t_depth=cost.t_depth)

def compose_k_sequential(cost, times, label="_"):
    if times == 0: return null_costf()
    assert cost.qubits_in == cost.qubits_out
    return LogicalCosts(label=label,
                        qubits_in=cost.qubits_in,
                        qubits_out=cost.qubits_out,
                        gates=cost.gates*times,
                        depth=cost.depth*times,
                        dw=cost.dw*times,
                        toffoli_count=cost.toffoli_count*times,
                        t_count=cost.t_count*times,
                        t_depth=cost.t_depth*times)

def compose_k_parallel(cost, times, label="_"):
    if times == 0: return null_costf()
    return LogicalCosts(label=label,
                        qubits_in=times * cost.qubits_in,
                        qubits_out=times * cost.qubits_out,
                        gates=times * cost.gates,
                        depth=cost.depth,
                        dw=times * cost.dw,
                        toffoli_count=times * cost.toffoli_count,
                        t_count=times * cost.t_count,
                        t_depth=cost.t_depth)

def compose_sequential(cost1, cost2, label="_"):
    assert cost1.qubits_out >= cost2.qubits_in
    # Pad unused wires with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.qubits_out > cost2.qubits_in:
        dw += (cost1.qubits_out - cost2.qubits_in) * cost2.depth
    return LogicalCosts(label=label,
                        qubits_in=cost1.qubits_in,
                        qubits_out=cost1.qubits_out - cost2.qubits_in + cost2.qubits_out,
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
                      gates=cost1.gates + cost2.gates,
                      depth=max(cost1.depth, cost2.depth),
                      dw=dw,
                      toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
                      t_count=cost1.t_count + cost2.t_count,
                      t_depth=max(cost1.t_depth, cost2.t_depth))



def classical_popcount_costf(n, k):
    ell = mp.ceil(mp.log(n,2)+1)
    t = mp.ceil(mp.log(k,2))
    gates = 10*n - 9*ell - t - 2
    depth = 1 + 2*ell + 2 + mp.ceil(mp.log(ell - t - 1, 2))

    cc = ClassicalCosts(label="popcount",
                        gates=gates,
                        depth=depth)

    return cc


def adder_costf(i, CI=False):
    """
    Logical cost of i bit adder (Cuccaro et al). With Carry Input if CI=True

    """
    # XXX: Check this!
    adder_cnots = 6 if i == 1 else (5*i+1 if CI else 5*i-3)
    adder_depth = 7 if i == 1 else (2*i+6 if CI else 2*i+4)
    adder_nots = 2*i-2 if CI else 2*i-4
    adder_tofs = 2*i-1
    adder_qubits_in = 2*i+1 if CI else 2*i
    adder_qubits_out = 2*i+2 if CI else 2*i+1
    adder_t_depth = adder_tofs * MagicConstants.t_depth_div_toffoli
    adder_t_count = adder_tofs * MagicConstants.t_div_toffoli
    adder_gates = adder_cnots + adder_nots + adder_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(label=str(i)+"-bit adder",
                        qubits_in=adder_qubits_in,
                        qubits_out=adder_qubits_out,
                        gates=adder_gates,
                        depth=adder_depth,
                        dw=adder_qubits_in * adder_depth,
                        toffoli_count=adder_tofs,
                        t_count=adder_t_count,
                        t_depth=adder_t_depth)


def hamming_wt_costf(n):
    """
    Logical cost of mapping |v>|0> to |v>|H(v)>.
    (The adder tree uses in-place addition, so some of the bits of |v> overlap |H(v)>
    and there are ancilla as well.)

    :param n: number of bits in v

    """
    b = int(mp.floor(log2(n)))
    if n == 1:
        qc = null_costf(qubits_in=1, qubits_out=1)
    elif n == 2:
        qc = adder_costf(1)
    elif bin(n+1).count('1') == 1:
        # n = 2**(b+1) - 1.
        # We can use every input bit of adder tree, including carry inputs.
        qc = null_costf(qubits_in=n,qubits_out=n)
        for i in range(1, b + 1):
            L = compose_k_parallel(adder_costf(i, CI=True), 2**(b-i))
            if L.qubits_in > qc.qubits_out:
              qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=L.qubits_in))
            qc = compose_sequential(qc, L)
    else:
        # Decompose n into a sum of terms of the form 2**i - 1
        qc = compose_parallel(hamming_wt_costf(2**b-1), hamming_wt_costf(n-(2**b-1)))
        adder = adder_costf(b)
        if adder.qubits_in > qc.qubits_out:
          qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=adder.qubits_out))
        qc = compose_sequential(qc, adder) # XXX: We could feed a bit to the carry input here.

    qc = compose_parallel(qc, null_costf(), label=str(n)+"-bit hamming weight")
    return qc


def carry_costf(m, c=None):
    """
    Logical cost of mapping |x> to (-1)^{(x+c)_m}|x> where
    (x+c)_m is the m-th bit (zero indexed) of x+c for an
    arbitrary m bit constant c.

    Numbers here are for CARRY circuit from Häner--Roetteler--Svore arXiv:1611.07995
    """
    # XXX: Might be cheaper to use "high bit only" variant of Cuccaro,
    # depending on optimisation metric.

    carry_qubits = m # XXX: assumes that there are m "dirty" ancilla available
    carry_toffoli_count = 4*(m-2) + 2
    carry_t_count = MagicConstants.t_div_toffoli * carry_toffoli_count
    carry_gates = MagicConstants.gates_div_toffoli * carry_toffoli_count # cnots
    carry_gates += 2 + 4*bin(int(2**m-c)).count('1')
    carry_depth = 3*carry_t_count # XXX
    carry_dw = carry_depth * (m+1) # XXX

    # XXX: Ignoring cost of initialising/discarding |-> for phase kickback

    return LogicalCosts(label="carry",
                      qubits_in=m,
                      qubits_out=m,
                      gates=carry_gates,
                      depth=carry_depth,
                      dw=carry_dw,
                      toffoli_count=carry_toffoli_count,
                      t_count=carry_t_count,
                      t_depth=carry_t_count)


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
    # XXX: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # XOR in the fixed sketch "u"
    # XXX: We're skipping ~ n NOT gates for mapping |v> to |u^v>
    qc = delay(qc, 1)

    # Use tree of adders compute hamming weight
    #     |i>|u^v_i>|0>     ->    |u^v_i>|wt(u^v_i)>
    hamming_wt = hamming_wt_costf(n)
    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires+hamming_wt.qubits_in))
    qc = compose_sequential(qc, hamming_wt)

    # Compute the high bit of (2^ceil(log(n)) - k) + hamming_wt
    #     |i>|v_i>|wt(u^v_i)>   ->     (-1)^popcnt(u,v_i) |u^v_i>|wt(u^v_i)>
    qc = compose_sequential(qc, carry_costf(int(mp.ceil(log2(n))), k))

    # Uncompute hamming weight.
    qc = compose_sequential(qc, reverse(hamming_wt))

    # Uncompute XOR
    # XXX: We're skipping ~ n NOT gates for mapping |u^v> to |v>
    qc = delay(qc, 1)

    # Uncompute table entry
    # XXX: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # Discard ancilla
    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires))

    qc = compose_parallel(qc, null_costf(), label="popcount"+str((n,k)))
    return qc


def diffusion_costf(L):
    """
    Logical cost of the diffusion operator D R_0 D^-1
    where
      D samples the uniform distribution on {1,...,L}
      R_0 is the unitary I - 2|0><0|

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    index_wires = int(mp.ceil(log2(L)))

    # a l-controlled NOT takes (32l - 84)T
    # the diffusion operator in our circuit is index_wires-controlled NOT
    # TODO: magic constants
    diffusion_t_count = 32 * index_wires - 84

    # Include hadamards on index wires to prepare uniform superposition
    # Ignore the cost of qRAM.
    diffusion_gates = diffusion_t_count + 2*index_wires

    # We currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    # XXX: Not clear that this is optimal.
    diffusion_t_depth = diffusion_t_count
    diffusion_depth = diffusion_t_depth + 2

    # +3 for one ancilla initialized in |->, targetted with cnot, then discarded
    diffusion_dw = index_wires * diffusion_depth + 3

    return LogicalCosts(label="diffusion",
                        qubits_in=index_wires,
                        qubits_out=index_wires,
                        gates=diffusion_gates,
                        depth=diffusion_depth,
                        dw=diffusion_dw,
                        toffoli_count=0, # XXX
                        t_count=diffusion_t_count,
                        t_depth=diffusion_t_depth)


def popcount_grover_iteration_costf(L, n, k):
    """
    Logical cost of G(popcount) = (D R_0 D^-1) R_popcount
    where
      D samples the uniform distribution on {1,...,L}
      (D R_0 D^-1) is the diffusion operator.
      R_popcount maps |i> to (-1)^{popcount(u,v_i)}|i> for some fixed u

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """

    popcount_cost = popcount_costf(L, n, k)
    diffusion_cost = diffusion_costf(L)

    return compose_sequential(diffusion_cost, popcount_cost, label="oracle")


def searchf(L, n, k):
    """
    Logical cost of popcount filtered search that succeeds w.h.p.
    where i and j are chosen so that ij ~ sqrt(L) * search_amplification_factor
    This is within a factor of 2 as long as the cost of S_ip < G(pc)^i

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
      # XXX: Double check that this is what we want.
      # quantum search does sqrt(1/positive_rate) popcounts
      # per inner product.
      return 1.0/positive_rate > MagicConstants.ip_div_pc**2


def all_pairs(d, n=None, k=None, epsilon=0.01, optimise=True, metric="t_count"):
    if n is None:
        n = 1
        while n < d:
            n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    pr = load_probabilities(d, n, k)

    epsilon = mp.mpf(epsilon)

    def cost(pr):
        L = (1+epsilon) * 2/((1-pr.eta)*C(pr.d,mp.pi/3))
        search_calls = int(mp.ceil(2*(1+epsilon)/(1-pr.eta) * L))
        expected_bucket_size = ((1-pr.eta)/(1+epsilon))**2 * L/2
        if metric == "G":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).gates
        elif metric == "DW":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).dw
        elif metric == "t_count":
            search_cost = searchf(expected_bucket_size, pr.n, pr.k).t_count
        elif metric == "naive_quantum":
            search_cost = mp.sqrt(expected_bucket_size)
        elif metric == "classical":
            search_cost = expected_bucket_size * classical_popcount_costf(pr.n, pr.k).gates
        elif metric == "naive_classical":
            search_cost = average_search_size
        else:
            raise ValueError("Unknown metric")
        return search_calls * search_cost

    positive_rate = pf(pr.d, pr.n, pr.k)
    while optimise and not popcounts_dominate_cost(positive_rate, metric):
        pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
        positive_rate = pf(pr.d, pr.n, pr.k)

    return pr.d, pr.n, pr.k, log2(cost(pr)), 1/positive_rate, metric


def random_buckets(d, n=None, k=None, theta1=None, optimise=True, metric="classical"):
    if n is None:
        n = 1
        while n < d:
            n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    theta = theta1 if theta1 else 1.2860
    pr = load_probabilities(d, n, k)
    # XXX: ip_cost is pretty arbitrary here
    ip_cost = MagicConstants.ip_div_pc * classical_popcount_costf(pr.n, pr.k).gates

    def cost(pr, T1):
        L = 2/((1-pr.eta)*C(pr.d, mp.pi/3))
        buckets = 1.0/W(pr.d, T1, T1, mp.pi/3)
        expected_bucket_size = L * C(pr.d, T1)
        fill_cost = L * ip_cost
        average_search_size = expected_bucket_size/2
        searches_per_bucket = expected_bucket_size
        if metric == "G":
            search_cost = searchf(average_search_size, pr.n, pr.k).gates
        elif metric == "DW":
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

    if optimise:
        theta = local_min(lambda T: cost(pr,T), theta, LOW=mp.pi/6, HIGH=mp.pi/2)
        # XXX: positive_rate is expensive to calculate
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, metric):
            pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
            theta = local_min(lambda T: cost(pr,T), theta, LOW=mp.pi/6, HIGH=mp.pi/2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return pr.d, pr.n, pr.k, theta, log2(cost(pr, theta)), metric


def table_buckets(d, n=None, k=None, theta1=None, theta2=None, optimise=True, metric="t_count"):
    if n is None:
        n = 1
        while n < d:
            n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    theta = theta1 if theta1 else mp.pi/3
    pr = load_probabilities(d, n, k)
    # XXX: ip_cost is pretty arbitrary here
    ip_cost = MagicConstants.ip_div_pc * classical_popcount_costf(pr.n, pr.k).gates

    def cost(pr, T1):
        T2 = T1 # TODO: Handle theta1 != theta2
        L = 2/((1-pr.eta)*C(d,mp.pi/3))
        search_calls = int(mp.ceil(L))
        filters = 1/W(d, T1, T2, mp.pi/3)
        populate_table_cost = L * filters * C(d,T2) * ip_cost
        relevant_bucket_cost = filters * C(d,T1) * ip_cost
        average_search_size = L * filters * C(d, T1) * C(d,T2) / 2
        # TODO: Scale insert_cost and relevant_bucket_cost?
        if metric == "G":
            search_cost = searchf(average_search_size, pr.n, pr.k).gates
        elif metric == "DW":
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

    if optimise:
        theta = local_min(lambda T: cost(pr,T), theta, LOW=mp.pi/6, HIGH=mp.pi/2)
        # XXX: positive_rate is expensive to calculate
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, metric):
            pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
            theta = local_min(lambda T: cost(pr,T), theta, LOW=mp.pi/6, HIGH=mp.pi/2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
          positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return pr.d, pr.n, pr.k, theta, theta, log2(cost(pr, theta)), metric

