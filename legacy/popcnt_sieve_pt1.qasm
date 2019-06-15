#
# File:   popcnt_sieve.qasm
# Date:   28-Nov-18
# Author: E. W. Postlethwaite
#
# Simple illustration of a circuit we might want to implement during quantum sieving
#
    defbox  add4,4,0,'\texttt{add}'

    qubit   \tilde{u}_{0}
    qubit   a0,0
    qubit   \tilde{v}_{0}
    qubit   \tilde{v}_{1}
    qubit   a1,0
    qubit   \tilde{u}_{1}
    qubit   \tilde{u}_{2}
    qubit   a2,0
    qubit   \tilde{v}_{2}
    qubit   \tilde{v}_{3}
    qubit   a3,0
    qubit   \tilde{u}_{3}

    cnot    \tilde{u}_{0},\tilde{v}_{0}
    cnot    \tilde{u}_{1},\tilde{v}_{1}
    cnot    \tilde{u}_{2},\tilde{v}_{2}
    cnot    \tilde{u}_{3},\tilde{v}_{3}

    add4    a0,\tilde{v}_{0},\tilde{v}_{1},a1
    add4    a2,\tilde{v}_{2},\tilde{v}_{3},a3
