# -*- coding: utf-8 -*-
"""
Constants to be dumped into the LaTeX file.
"""
import os
import csv

from mpmath import mp
from probabilities import C
from cost import list_decoding, log2, load_probabilities


def p(name, val):
    if type(val) is int:
        print("/consts/{:s}/.initial={:d},".format(name, val))
    elif type(val) is str:
        print("/consts/{:s}/.initial={:s},".format(name, val))
    else:
        print("/consts/{:s}/.initial={:.1f},".format(name, val))

def load_csv(f, metric):
    filename = os.path.join("..", "data", "cost-estimate-{f}-{metric}.csv")
    filename = filename.format(f=f, metric=metric)
    with open(filename, "r") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=",", quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL)
        D = {int(L['d']) : L for L in csvreader}
    return D

def main():
    data_bdgl_dw = load_csv("list_decoding", "dw")
    data_bdgl_ge19 = load_csv("list_decoding", "ge19")
    data_bdgl_classical = load_csv("list_decoding", "classical")
    data_size_bits = load_csv("sieve_size", "bits")

    #p("classical784", float(list_decoding(784, metric="classical").log_cost))
    #p("classical1024", float(list_decoding(1024, metric="classical").log_cost))

    for d in sorted(data_bdgl_ge19):
        if float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_ge19[d]['log_cost']) > 0:
            p("bdgl/ge19/crossover", d)
            break

    for d in sorted(data_bdgl_ge19):
        if float(data_bdgl_ge19[d]['log_cost']) > 128:
            p("bdgl/ge19/dim/cost128", d)
            p("bdgl/ge19/adv/cost128", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_ge19[d]['log_cost']))
            break

    for d in sorted(data_bdgl_ge19):
        if float(data_bdgl_ge19[d]['log_cost']) > 256:
            p("bdgl/ge19/dim/cost256", d)
            p("bdgl/ge19/adv/cost256", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_ge19[d]['log_cost']))
            break

    for d in sorted(data_bdgl_dw):
        if float(data_bdgl_dw[d]['log_cost']) > 128:
            p("bdgl/dw/dim/cost128", d)
            p("bdgl/dw/adv/cost128", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_dw[d]['log_cost']))
            break

    for d in sorted(data_bdgl_dw):
        if float(data_bdgl_dw[d]['log_cost']) > 256:
            p("bdgl/dw/dim/cost256", d)
            p("bdgl/dw/adv/cost256", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_dw[d]['log_cost']))
            break

    for d in sorted(data_size_bits):
        if d % 16 == 0 and float(data_size_bits[d]['log2_size']) > 127:
            p("size128/dim", d)
            p("bdgl/ge19/adv/size128", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_ge19[d]['log_cost']))
            p("bdgl/dw/adv/size128", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_dw[d]['log_cost']))
            break

    # Assume a moon sized memory with 1 petabyte per gram density
    for d in sorted(data_size_bits):
        if d % 16 == 0 and float(data_size_bits[d]['log2_size']) > 139:
            p("size140/dim", d)
            p("bdgl/ge19/adv/size140", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_ge19[d]['log_cost']))
            p("bdgl/dw/adv/size140", float(data_bdgl_classical[d]['log_cost']) - float(data_bdgl_dw[d]['log_cost']))
            break


if __name__ == "__main__":
    main()
