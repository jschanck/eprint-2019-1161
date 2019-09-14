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
    gates_div_toffoli = 17  # 7 T, 7 CNOT, 2 H, 1 S

    AMMR12_tof_t_count = 7
    AMMR12_tof_t_depth = 3
    AMMR12_tof_gates   = 17
    AMMR12_tof_depth   = 10

    """
    We will accept a list size growth by this factor.
    """

    list_growth_bound = 2.0

    """
    Assuming 32 bits are used to represent full vectors, we expect the ratio
    between a full inner product and a popcount call to be 32^2 * d / n, where
    32^2 is the cost of a naive multiplier and n approximates the cost of a
    hamming weight call.
    """

    word_size = 32

    """
    In a filtered quantum search we nest 'Grover search with an unknown number
    of marked elements' inside of 'amplitude amplification with an unknown
    success probability'.  We numerically optimise the trade-off between the
    amount of amplification (number of queries) and success probability
    (expected number of repetitions).  q(x) = 3*asin(sqrt(x))/(Pi +
    asin(sqrt(x)) search_amplification_factor = 1/sqrt(x)
    search_repetition_factor = 1/(1-q(x))
    """
    filter_amplification_factor = 1.67199  # x = 0.3577070585
    filter_repetition_factor    = 2.03439  # x = 0.3577070585
