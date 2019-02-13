#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#
    defbox  add4,4,0,'\texttt{add}'

    qubit   u0
    qubit   a0,0
    qubit   v0
    qubit   v1
    qubit   a1,0
    qubit   u1
    qubit   u2
    qubit   a2,0
    qubit   v2
    qubit   v3
    qubit   a3,0
    qubit   u3

    cnot    u0,v0
    cnot    u1,v1
    cnot    u2,v2
    cnot    u3,v3

    add4    a0,v0,v1,a1
    add4    a2,v2,v3,a3
