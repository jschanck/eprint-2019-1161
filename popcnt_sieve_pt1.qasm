#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#
    defbox  add4,4,0,'\texttt{add}'

    qubit   v0
    qubit   a0,0
    qubit   w0
    qubit   w1
    qubit   a1,0
    qubit   v1
    qubit   v2
    qubit   a2,0
    qubit   w2
    qubit   w3
    qubit   a3,0
    qubit   v3

    cnot    v0,w0
    cnot    v1,w1
    cnot    v2,w2
    cnot    v3,w3

    add4    a0,w0,w1,a1
    add4    a2,w2,w3,a3
