#
# File:   nexclusive_or.qasm
# Date:   29-Nov-2018
# Author: E. W. Postlethwaite
#
# A non exclusive reversible OR gate. If a is instead 1, then a performs NOT non exclusive OR
#

    qubit   a,0
    qubit   x
    qubit   y

    toffoli x,y,a
    cnot    y,x
    cnot    a,x
