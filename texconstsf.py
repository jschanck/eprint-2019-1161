# -*- coding: utf-8 -*-
"""
Constants to be dumped into the LaTeX file.
"""
from mpmath import mp
from probabilities import C
from cost import list_decoding, log2, load_probabilities


def main():
    def p(name, val):
        if type(val) is int:
            print("/consts/{:s}/.initial={:d},".format(name, val))
        elif type(val) is str:
            print("/consts/{:s}/.initial={:s},".format(name, val))
        else:
            print("/consts/{:s}/.initial={:.1f},".format(name, val))

    d = 256
    while float(list_decoding(d + 16, metric="classical").log_cost - list_decoding(d + 16, metric="ge19").log_cost) < 0:
        d += 16
        p("ge19crossover", d)
        p(
            "ge19adv512",
            float(list_decoding(512, metric="classical").log_cost - list_decoding(512, metric="ge19").log_cost),
        )
        p(
            "ge19adv768",
            float(list_decoding(768, metric="classical").log_cost - list_decoding(768, metric="ge19").log_cost),
        )
        p(
            "ge19adv1024",
            float(list_decoding(1024, metric="classical").log_cost - list_decoding(1024, metric="ge19").log_cost),
        )

    d = 352
    p("real/dim", int(d))
    xc = list_decoding(d, metric="classical")
    xdw = list_decoding(d, metric="dw")
    p("real/ram", float(xc.log_cost))
    p("real/dw", float(xdw.log_cost))
    p("real/adv", float(log2(2 ** xc.log_cost / 2 ** xdw.log_cost)))
    p("real/one/dw", float(log2(xdw.detailed_costs.dw)))
    p("real/one/depth", float(log2(xdw.detailed_costs.depth)))
    p("real/one/width", float(log2(xdw.detailed_costs.qubits_max)))
    p("real/one/ram", float(log2(xc.detailed_costs.gates)))
    p("real/one/ramdepth", float(log2(xc.detailed_costs.depth)))

    md = 96

    pr = load_probabilities(xdw.d, xdw.n, xdw.k)
    tot = log2(2 / ((1 - pr.eta) * C(d, mp.pi / 3)))
    seq = md - log2(xdw.detailed_costs.depth)
    qpar = tot - seq
    p("real/md96/qpar", float(qpar))
    p("real/md96/qubits", float(log2(2 ** qpar * xdw.detailed_costs.qubits_max)))

    pr = load_probabilities(xc.d, xc.n, xc.k)
    tot = log2(2 / ((1 - pr.eta) * C(d, mp.pi / 3)))
    seq = md - log2(xc.detailed_costs.depth)
    cpar = tot - seq
    p("real/md96/cpar", float(cpar))


if __name__ == "__main__":
    main()
