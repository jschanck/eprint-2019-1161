#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#

    qubit   \tilde{v}_{1}
    qubit   \tilde{v}_{3}
    qubit   s^{23}_{0}
    qubit   s^{23}_{1}
    qubit   s^{0123}_{0}
    qubit   a0
    qubit   s^{0123}_{1}
    qubit   s^{0123}_{2}
    qubit   \tilde{u}_{0}
    qubit   \tilde{u}_{1}
    qubit   \tilde{u}_{2}
    qubit   \tilde{u}_{3}

    toffoli s^{0123}_{1},s^{0123}_{2},a0
    cnot    s^{0123}_{2},s^{0123}_{1}
    cnot    a0,s^{0123}_{1}

    nop     \tilde{v}_{1}
