#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#
    defbox  add6,6,0,'\texttt{add}'

    qubit   w1
    qubit   w3
    qubit   a0
    qubit   s^{01}_{0}
    qubit   s^{23}_{0}
    qubit   s^{01}_{1}
    qubit   s^{23}_{1}
    qubit   a2
    qubit   v0
    qubit   v1
    qubit   v2
    qubit   v3

    add6    a0,s^{01}_{0},s^{23}_{0},s^{01}_{1},s^{23}_{1},a2

    nop     w1
