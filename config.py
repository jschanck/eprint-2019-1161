# -*- coding: utf-8 -*-
class MagicConstants:

    """
    k/n ≈ 1/3 and chosen to keep false negative rate ≤ 2.
    """

    k_div_n = 11/32.

    """
    We assume one Toffoli gate takes 7 T-gates.
    """

    t_div_toffoli = 7
    t_depth_div_toffoli = 3
    gates_div_toffoli = 17 # 7 T, 7 CNOT, 2 H, 1 S

    """
    We will accept a list size growth by this factor.
    """

    list_growth_bound = 2.0

    """
    We somewhat arbitrarily assume that an inner product is at most this much more expensive than a
    pocount test.
    """

    ip_div_pc = 1000

    """
    In a filtered quantum search we nest 'Grover search with an unknown
    number of marked elements' inside of 'amplitude amplification with
    an unknown success probability'. We numerically optimise the trade-off
    between the amount of amplification (number of queries) and success
    probability (expected number of repetitions).
    q(x) = 3*asin(sqrt(x))/(Pi + asin(sqrt(x))
    search_amplification_factor = 1/sqrt(x)
    search_repetition_factor = 1/(1-q(x))
    """
    #filter_amplification_factor = 1.70130 # x = 0.3454915028
    #filter_repetition_factor    = 2.00000 # x = 0.3454915028
    filter_amplification_factor = 1.67199 # x = 0.3577070585
    filter_repetition_factor    = 2.03439 # x = 0.3577070585

