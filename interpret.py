#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from sage.all import var, find_fit, log, latex, floor
import csv

C_wrapper = list(csv.reader(open("../data/wrapper.csv", "r")))[1:]
C_g6k = list(csv.reader(open("../data/g6k.csv", "r")))[1:]


def curve_fit(D, include_log=False):
    a, b, d = var("a,b,d")
    if include_log:
        f = a*d + log(d, 2) + b
    else:
        f = a*d + b
    return f.subs(find_fit(D, f.function(d), solution_dict=True))


quantum_gauss_real = curve_fit([(c[0], c[1]) for c in C_wrapper[len(C_wrapper)//2:]])
quantum_gauss_spec = curve_fit([(c[0], c[3]) for c in C_wrapper[len(C_wrapper)//2:]])
classical_gauss    = curve_fit([(c[0], log(float(c[1])*2.0*10**9, 2)) for c in C_g6k[10:]], include_log=True)


def output_curve(f, shift=0):
    x = var("x")
    f = f + shift
    plot = (r"\addplot+ [solid,thin,black,domain=10:550]{%s};"%(f.subs(d=x))).replace("log(x)/log(2)", "log2(x)")
    legend = (r"\addlegendentry{\(%s\)}"%(latex(f))
              .replace("\\frac{\\log\\left(d\\right)}{\\log\\left(2\\right)}", "\\log_2(d)"))
    return "\n".join([plot, legend, ""])


def crossover(f, g, f_shift=0, g_shift=0):
    d = var("d")
    return floor((f + f_shift == g + g_shift).solve(d)[0].rhs())


d = var("d")
print output_curve(quantum_gauss_real, log(d, 2) + 10)
print output_curve(quantum_gauss_spec, log(d, 2) + 10)
print output_curve(classical_gauss)
print crossover(quantum_gauss_real, classical_gauss, f_shift=log(d, 2) + 10)
print crossover(quantum_gauss_spec, classical_gauss, f_shift=log(d, 2) + 10)
