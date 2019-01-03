#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#

    qubit   v0
    qubit   v1
    qubit   v2
    qubit   v3
    qubit   s^{0123}_{0}
    qubit   a0
    qubit   s^{0123}_{1}
    qubit   s^{0123}_{2}
    qubit   w1
    qubit   w3
    qubit   s^{23}_{0}
    qubit   s^{23}_{1}

    toffoli s^{0123}_{1},s^{0123}_{2},a0
    cnot    s^{0123}_{2},s^{0123}_{1}
    cnot    a0,s^{0123}_{1}

    nop     v0
