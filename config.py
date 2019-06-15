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
