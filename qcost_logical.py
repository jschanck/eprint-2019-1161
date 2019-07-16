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
                           "t_count", "t_depth", "t_width"))

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

def popcounts_dominate_cost(positive_rate, metric):
  if metric in ClassicalMetrics:
    return 1.0/positive_rate > MagicConstants.ip_div_pc
  else:
    # XXX: Double check that this is what we want.
    # quantum search does sqrt(test_ratio) popcounts
    # per inner product.
    return 1.0/positive_rate > MagicConstants.ip_div_pc**2

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

    popcount_qubits = index_wires # XXX: Include other qubits
    popcount_gates = popcount_cnots + popcount_tofs # XXX: Include other gates

    qc = LogicalCosts(label="popcount",
                      params=(L, n, k),
                      qubits=popcount_qubits,
                      gates=popcount_gates,
                      dw=popcount_qubits * popcount_t_depth, # XXX
                      toffoli_count=popcount_tofs,
                      t_count=popcount_t_count,
                      t_depth=popcount_t_depth,
                      t_width=popcount_t_count/popcount_t_depth)
    return qc


def classical_popcount_costf(n, k):
  ell = mp.ceil(mp.log(n,2)+1)
  t = mp.ceil(mp.log(k,2))
  gates = 10*n - 9*ell - t - 2
  depth = 1 + 2*ell + 2 + mp.ceil(mp.log(ell - t - 1, 2))

  cc = ClassicalCosts(label="popcount",
                      gates=gates,
                      depth=depth)

  return cc


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
                        qubits=None,
                        gates=None,
                        dw=None,
                        toffoli_count=None,
                        t_count=t_count, t_depth=t_depth, t_width=t_width)


def searchf(L, n, k):
    """
    Logical cost of finding a marked element in a list

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    L, n, k, index_wires = _preproc_params(L, n, k)

    oracle_cost = oracle_costf(L, n, k)
    oracle_calls = mp.sqrt(L)
    oracle_calls *= MagicConstants.search_amplification_factor
    oracle_calls *= MagicConstants.search_repetition_factor

    t_count = oracle_calls * oracle_cost.t_count
    t_depth = oracle_calls * oracle_cost.t_depth
    t_width = oracle_cost.t_width

    qc = LogicalCosts(label="search",
                        params=(L, n, k),
                        qubits=None,
                        gates=None,
                        dw=None,
                        toffoli_count=None,
                        t_count=t_count, t_depth=t_depth, t_width=t_width)

    return qc

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
      #L = 2/((1-pr.eta)*C(d,mp.pi/3))
      L = 1/C(d,mp.pi/3)
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

    return pr.d, pr.n, pr.k, theta, log2(cost(pr, theta)), 1/positive_rate, metric

