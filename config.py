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
    """
    #search_amplification_factor = 1.7013  # 1/sqrt(0.3454915028)
    #search_repetition_factor    = 8.0000  # 4/(1-(3*asin(sqrt(0.3454915028)) / (Pi + asin(sqrt(0.3454915028)))))
    search_amplification_factor = 1.6719 # 1/sqrt(0.3577070585)
    search_repetition_factor    = 8.1376 # 4/(1-(3*asin(sqrt(0.3577070585)) / (Pi + asin(sqrt(0.3577070585)))))
