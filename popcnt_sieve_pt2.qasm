#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#
    defbox  add6,6,0,'\texttt{add}'

    qubit   v1
    qubit   v3
    qubit   a0
    qubit   s^{01}_{0}
    qubit   s^{23}_{0}
    qubit   s^{01}_{1}
    qubit   s^{23}_{1}
    qubit   a2
    qubit   u0
    qubit   u1
    qubit   u2
    qubit   u3

    add6    a0,s^{01}_{0},s^{23}_{0},s^{01}_{1},s^{23}_{1},a2

    nop     v1
