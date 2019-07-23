#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mpmath import mp
from collections import namedtuple
from utils import load_probabilities
from config import MagicConstants
from probabilities_estimates import W, C, pf


LogicalCosts = namedtuple("LogicalCosts",
                          ("label", "params",
                           "qubits", "gates", "dw", "toffoli_count",
                           "t_count", "t_depth"))

PhysicalCosts = namedtuple("PhysicalCosts", ("label", "physical_qubits", "surface_code_cycles")) # What else?

ClassicalCosts = namedtuple("ClassicalCosts", ("label", "gates", "depth")) # What else?

ClassicalMetrics = {"classical", "naive_classical"}
QuantumMetrics = {"G", "DW", "t_count", "naive_quantum"}
Metrics = ClassicalMetrics | QuantumMetrics


def log2(x):
    return mp.log(x)/mp.log(2)


def local_min(f,x,D1=2,D2=5):
    y = f(x)
    for k in range(D1, D2+1):
        d = 0.1**k
        y2 = f(x+d)
        if y2 > y:
            d = -d
            y2 = f(x+d)
        while y2 < y:
            y = y2
            x = x+d
            y2 = f(x+d)
    return x


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


def null_costf():
    return LogicalCosts(label="null", params=None,
                        qubits=0, gates=0, dw=0, toffoli_count=0,
                        t_count=0, t_depth=0)

def compose_k_sequential(cost, times, label="_"):
    if times == 0: return null_costf()
    return LogicalCosts(label=label,
                      params=cost.params,
                      qubits=cost.qubits,
                      gates=cost.gates*times,
                      dw=cost.dw*times,
                      toffoli_count=cost.toffoli_count*times,
                      t_count=cost.t_count*times,
                      t_depth=cost.t_depth*times)

def compose_k_parallel(cost, times, label="_"):
    if times == 0: return null_costf()
    return LogicalCosts(label=label,
                      params=cost.params,
                      qubits=times * cost.qubits,
                      gates=times * cost.gates,
                      dw=times * cost.dw,
                      toffoli_count=times * cost.toffoli_count,
                      t_count=times * cost.t_count,
                      t_depth=cost.t_depth)

def compose_sequential(cost1, cost2, label="_"):
    return LogicalCosts(label=label,
                      params=cost1.params,
                      qubits=max(cost1.qubits, cost2.qubits),
                      gates=cost1.gates + cost2.gates,
                      dw=cost1.dw + cost2.dw,
                      toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
                      t_count=cost1.t_count + cost2.t_count,
                      t_depth=cost1.t_depth + cost2.t_depth)


def compose_parallel(cost1, cost2, label="_"):
    return LogicalCosts(label=label,
                      params=cost1.params,
                      qubits=cost1.qubits + cost2.qubits,
                      gates=cost1.gates + cost2.gates,
                      dw=cost1.dw + cost2.dw,
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
    adder_cnots = 6 if i == 1 else (5*i+1 if CI else 5*i-3)
    adder_depth = 7 if i == 1 else (2*i+6 if CI else 2*i+4)
    adder_nots = 2*i-2 if CI else 2*i-4
    adder_tofs = 2*i-1
    adder_qubits = 2*i+2
    adder_t_depth = adder_tofs * MagicConstants.t_depth_div_toffoli
    adder_t_count = adder_tofs * MagicConstants.t_div_toffoli
    adder_gates = adder_cnots + adder_nots + adder_tofs * MagicConstants.gates_div_toffoli
    return LogicalCosts(label=str(i)+"-bit adder",
                        params=None,
                        qubits=adder_qubits,
                        gates=adder_gates,
                        dw=adder_qubits * adder_depth,
                        toffoli_count=adder_tofs,
                        t_count=adder_t_count,
                        t_depth=adder_t_depth)


def hamming_wt_costf(n):
    """
    Logical cost of mapping |v>|0> to |v>|H(v)>

    :param n: number of bits in v

    """
    qc = null_costf()
    b = int(mp.floor(log2(n)))
    n = n - 2**b
    for i in range(1, b + 1):
        if n > 0: # Feed some bits to carry inputs
          L = compose_parallel(
                compose_k_parallel(adder_costf(i, CI=True),            min(n, 2**(b-i))),
                compose_k_parallel(adder_costf(i),          2**(b-i) - min(n, 2**(b-i))))
          n = n - min(n, 2**(b-i))
        else:
          L = compose_k_parallel(adder_costf(i), 2**(b-i))
        qc = compose_sequential(qc, L)
    qc = compose_sequential(qc, null_costf(), label=str(n)+"-bit hamming weight")
    return qc


def carry_costf(m, c=None):
    """
    Logical cost of mapping |x>|0> to |x>|(x+c)_m> where
    (x+c)_m is the m-th bit (zero indexed) of x+c for an
    arbitrary m bit constant c.

    Numbers here are for CARRY circuit from Häner--Roetteler--Svore arXiv:1611.07995
    """
    carry_qubits = m+1 # XXX: assumes that there are m "dirty" ancilla available
    carry_toffoli_count = 4*(m-2) + 2
    carry_t_count = MagicConstants.t_div_toffoli * carry_toffoli_count
    carry_gates = MagicConstants.gates_div_toffoli * carry_toffoli_count # cnots
    carry_gates += 2 + 4*bin(int(2**m-c)).count('1')
    carry_depth = 3*carry_t_count # XXX
    carry_dw = carry_depth * (m+1) # XXX

    return LogicalCosts(label="carry",
                      params=None,
                      qubits=carry_qubits,
                      gates=carry_gates,
                      dw=carry_dw,
                      toffoli_count=carry_toffoli_count,
                      t_count=carry_t_count,
                      t_depth=carry_t_count)


def popcount_costf(L, n, k):
    """
    Logical cost of mapping |v> to (-1)^{popcount(u,v)}|v> for fixed u.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """

    # XXX: We're skipping ~ n NOT gates for mapping |v> to |u^v>

    # Use tree of adders compute hamming weight
    #     |u^v>|0>     ->    |u^v>|wt(u^v)>
    hamming_wt_cost = hamming_wt_costf(n)

    # Compute the high bit of (2^ceil(log(n)) - k) + hamming_wt
    #     |v>|wt(u^v)>|->   ->     (-1)^popcnt(u,v) |u^v>|wt(u^v)>|->
    carry_cost = carry_costf(mp.ceil(log2(n)), k)
    qc = compose_sequential(hamming_wt_cost, carry_cost)

    # Uncompute hamming weight.
    #    (-1)^popcnt(u,v) |u^v>|wt> -> (-1)^popcnt(u,v) |u^v>|0>
    qc = compose_sequential(qc, hamming_wt_cost)

    # XXX: We're skipping ~ n NOT gates for mapping |v> to |u^v>

    return qc


def diffusion_costf(L, n, k):
    """
    Logical cost of the diffusion operator D S_0 D^-1
    where
      D samples the uniform distribution on a set of size L
      S_0 is the unitary I - 2|0><0|

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    L, n, k, index_wires = _preproc_params(L, n, k)

    # index_wires for address, n for |v>, 1 in |-> for phase kickback
    diffusion_qubits = index_wires + n + 1

    # a l-controlled NOT takes (32l - 84)T
    # the diffusion operator in our circuit is (index_wires - 1)-controlled NOT
    # TODO: magic constants
    # XXX: Shouldn't this be an n-controlled not?
    diffusion_t_count = 32 * (index_wires - 1) - 84

    # Include hadamards on index wires to prepare uniform superposition
    # Ignore the cost of qRAM.
    diffusion_gates = diffusion_t_count + 2*index_wires

    # We currently make an assumption (favourable to a sieving adversary) that the T gates in the
    # diffusion operator are all sequential and therefore bring down the average T gates required
    # per T depth.
    # XXX: Not clear that this is optimal.
    diffusion_t_depth = diffusion_t_count

    diffusion_dw = diffusion_qubits * (diffusion_t_depth + 2)

    return LogicalCosts(label="diffusion",
                       params=(L, n, k),
                       qubits=diffusion_qubits,
                       gates=diffusion_gates,
                       dw=diffusion_dw,
                       toffoli_count=0,
                       t_count=diffusion_t_count,
                       t_depth=diffusion_t_depth,
                       t_width=diffusion_t_count/diffusion_t_depth)


def oracle_costf(L, n, k):
    """
    Logical cost of G(popcount) = (D S_0 D^-1) S_popcount
    where
      D samples the uniform distribution on a set of size L
      S_0 is the unitary I - 2|0><0|
      S_popcount maps |v> to (-1)^{popcount(u,v)}|v> for some fixed u

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    popcount_cost = popcount_costf(L, n, k)
    diffusion_cost = diffusion_costf(L, n, k)

    return compose_sequential(diffusion_cost, popcount_cost, label="oracle")


def searchf(L, n, k):
    """
    Logical cost of popcount filtered search that succeeds w.h.p.
      G(G(pc)^i D, ip)^j =
        ((G(pc)^i D) S_0 (G(pc)^i D)^-1 S_ip)^j G(pc)^i D
    where i and j are chosen so that ij ~ sqrt(L) * search_amplification_factor
    We only cost G(pc)^ij and G(pc)^-ij.
    This is within a factor of 2 as long as the cost of S_ip < G(pc)^i

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    oracle_cost = oracle_costf(L, n, k)

    # The search routine takes two integer parameters m1 and m2.
    # We pick a random number in {0, ..., m1-1} and another in {0, ..., m2-1}.
    # We use an expected m1/2 iterations of the popcount oracle for the sampling
    # routine in amplitude amplification. Hence an expected m1 popcount oracles per
    # AA iteration. Assuming m1 > pi/4 * sqrt(L/P), where P is the number of popcount
    # positives. Since we don't know P exactly, we're going to leave some probability mass
    # on popcount negatives. We have to account for that in the AA step. We do an expected
    #     m2 * filter_amplification_factor / 2 AA iterations.
    # If we assume m2 = pi/4 * sqrt(P)
    # then the whole thing succeeds with probability 1/2 * 1/filter_repetition factor.
    # So in we'll repeat 2*filter_repetition factor times.
    # We do m1*m2 total oracle calls.
    # Assume best case for adversary, m1=pi/4 * sqrt(L/P) and m2=pi/4*sqrt(P) then
    # this is (pi/4)^2 * filter_amplification_factor * filter_repetition_factor * sqrt(L)
    # popcount oracle calls. This is about 2 * sqrt(L) popcount oracle calls.

    oracle_calls  = mp.sqrt(L)
    oracle_calls *= mp.pi*mp.pi/16
    oracle_calls *= MagicConstants.filter_amplification_factor
    oracle_calls *= MagicConstants.filter_repetition_factor

    return compose_k_sequential(oracle_cost, oracle_calls, label="search")


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

    def cost(pr, T1):
      L = 2/((1-pr.eta)*C(pr.d, mp.pi/3))
      buckets = 1.0/W(pr.d, T1, T1, mp.pi/3)
      expected_bucket_size = L * C(pr.d, T1)
      fill_cost = L # XXX: Refine cost estimate?
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
      theta = local_min(lambda T: cost(pr,T), theta)
      # XXX: positive_rate is expensive to calculate
      positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
      while not popcounts_dominate_cost(positive_rate, metric):
        pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
        theta = local_min(lambda T: cost(pr,T), theta)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
      positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return pr.d, pr.n, pr.k, theta, log2(cost(pr, theta)), 1/positive_rate, metric


def table_buckets(d, n=None, k=None, theta1=None, theta2=None, optimise=True, metric="t_count"):
    if n is None:
      n = 1
      while n < d:
        n = 2*n

    k = k if k else int(MagicConstants.k_div_n * n)
    theta = theta1 if theta1 else mp.pi/3
    pr = load_probabilities(d, n, k)

    def cost(pr, T1):
      T2 = T1 # TODO: Handle theta1 != theta2
      L = 2/((1-pr.eta)*C(d,mp.pi/3))
      search_calls = int(mp.ceil(L))
      filters = 1/W(d, T1, T2, mp.pi/3)
      populate_table_cost = L * filters * C(d,T2)
      relevant_bucket_cost = filters * C(d,T1)
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
      theta = local_min(lambda T: cost(pr,T), theta)
      # XXX: positive_rate is expensive to calculate
      positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
      while not popcounts_dominate_cost(positive_rate, metric):
        pr = load_probabilities(pr.d, 2*pr.n, int(MagicConstants.k_div_n * 2 * pr.n))
        theta = local_min(lambda T: cost(pr,T), theta)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
      positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    return pr.d, pr.n, pr.k, theta, theta, log2(cost(pr, theta)), 1/positive_rate, metric

