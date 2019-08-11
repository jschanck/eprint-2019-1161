# -*- coding: utf-8 -*-
from mpmath import mp
from collections import OrderedDict
from probabilities_estimates import probabilities, Probabilities
from config import MagicConstants

import os
import cPickle


class PrecomputationRequired(Exception):
    pass


def pretty_probs(probs, dps=10):
    """
    Take a ``Probabilities`` object and pretty print the estimated probabilities.

    :param probs: a ``Probabilitiess`` object.

    """
    fmt = "{0:7s}: {1:%ds}"%dps
    with mp.workdps(dps):
        print(fmt.format("gr", probs.gr))
        print(fmt.format("ngr", 1 - probs.gr))
        print(fmt.format("pf", probs.pf))
        print(fmt.format("npf", 1 - probs.pf))
        print(fmt.format("gr^pf", probs.gr_pf))
        print(fmt.format("ngr^pf", probs.ngr_pf))
        print(fmt.format("gr|pf", probs.gr_pf/probs.pf))
        print(fmt.format("pf|gr", probs.gr_pf/probs.gr))
        print(fmt.format("ngr|pf", probs.ngr_pf/probs.pf))


def create_bundle(d, n, K=None, BETA=None, prec=None):
    """
    Create a bundle of probabilities.

    :param d: We consider the sphere `S^{d-1}`.
    :param n: Number of popcount vectors.
    :param K: We consider all `k ∈ K` as popcount thresholds (default `k = 5/16⋅n`).
    :param BETA: We consider all caps parameterized by `β in BETA` (default: No cap).
    :param prec: We compute with this precision (default: 53).

    """
    bundle = OrderedDict()

    prec = prec if prec else mp.prec
    BETA = BETA if BETA else (None,)
    K = K if K else (int(MagicConstants.k_div_n*n),)

    if 2**mp.floor(mp.log(n, 2)) != n:
        raise ValueError("n must be a power of two but got %d"%n)

    for k in K:
        if not 0 <= k <= n:
            raise ValueError("k not in [0, %d]"%(0, n))

    for beta in BETA:
        beta_mpf = mp.mpf(beta) if beta else None
        beta_flt = float(beta) if beta else None
        for k in K:
            bundle[(d, n, k, beta_flt)] = probabilities(d, n, k, beta=beta_mpf, prec=prec)

    return bundle


def bundle_fn(d, n=None):
    if n is None:
        d, n = [keys[:2] for keys in d.keys()][0]
    return os.path.join("probabilities", "%03d_%04d"%(d, n))


def store_bundle(bundle):
    """
    Store a bundle in a flat format for compatibility reasons.

    In particular, mpf values are converted to strings.

    """
    bundle_ = OrderedDict()

    for (d, n, k, beta) in bundle:
        with mp.workprec(bundle[(d, n, k, beta)].prec):
            vals = OrderedDict([(k_, str(v_)) for k_, v_ in bundle[(d, n, k, beta)].__dict__.items()])
        bundle_[(d, n, k, beta)] = vals

    with open(bundle_fn(bundle), "wb") as fh:
        cPickle.dump(bundle_, fh)


def load_bundle(d, n, compute=False):
    """
    Load bundle from the flat format and convert into something we can use.

    """
    bundle = OrderedDict()
    try:
        with open(bundle_fn(d, n), "rb") as fh:
            bundle_ = cPickle.load(fh)
            for (d, n, k, beta) in bundle_:
                with mp.workprec(int(bundle_[(d, n, k, beta)]["prec"])):
                    d_ = dict()
                    for k_, v_ in bundle_[(d, n, k, beta)].items():
                        if "." in v_:
                            v_ = mp.mpf(v_)
                        elif v_ == "None":
                            v_ = None
                        else:
                            v_ = int(v_)
                        d_[k_] = v_
                    bundle[(d, n, k, beta)] = Probabilities(**d_)
            return bundle
    except IOError:
        if compute:
            return create_bundle(d, n, prec=int(compute))
        else:
            raise PrecomputationRequired()


def __bulk_create_and_store_bundles(args):
    d, n, BETA, prec = args
    bundle = create_bundle(d, n, BETA=BETA, prec=prec)
    store_bundle(bundle)


def bulk_create_and_store_bundles(D,
                                  N=(128, 256, 512, 1024, 2048, 4096, 8192),
                                  BETA=(None, mp.pi/3-mp.pi/10, mp.pi/3, mp.pi/3+mp.pi/10),
                                  prec=2*53, ncores=1):
    """
    Precompute a bunch of probabilities.
    """
    from multiprocessing import Pool
    jobs = []
    for d in D:
        for n in N:
            jobs.append((d, n, BETA, prec))

    if ncores > 1:
        return list(Pool(ncores).imap_unordered(__bulk_create_and_store_bundles, jobs))
    else:
        return map(__bulk_create_and_store_bundles, jobs)


def sanity_check_probabilities(probs):
    from sieves import cnkf

    if cnkf(probs) > MagicConstants.list_growth_bound:
        raise ValueError("List growth (%.1f) beyond tolerable limits (%.1f)"%(float(cnkf(probs)),
                                                                              MagicConstants.list_growth_bound))
    if 1/probs.pf < MagicConstants.ip_div_pc:
        raise ValueError("The cost of inner products might dominate for these parameters")


def load_probabilities(d, n, k, beta=None, compute=False, sanity_check=False):
    probs = load_bundle(d, n, compute=compute)[(d, n, k, beta)]
    if sanity_check:
        sanity_check_probabilities(probs)
    return probs


def stats(d, n, k, beta=None, prec=None):
    """
    Useful quantites.

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    from probabilities_estimates import C

    probs = probabilities(d, n, k, beta=beta, prec=prec)
    N = 1/C(d, mp.pi/3)
    P = probs.pf*N
    ckn = 1/(1-probs.eta)
    return (N, P, ckn)
