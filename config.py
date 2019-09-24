# -*- coding: utf-8 -*-


class MagicConstants:

    """
    k/n ≈ 1/3.
    """

    k_div_n = 1/3.

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
    Assuming 32 bits are used to represent full vectors, we expect the ratio
    between a full inner product and a popcount call to be 32^2 * d / n, where
    32^2 is the cost of a naive multiplier and n approximates the cost of a
    hamming weight call.
    """

    word_size = 32
