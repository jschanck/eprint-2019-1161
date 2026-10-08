# -*- coding: utf-8 -*-
"""
Estimating relevant probabilities on the sphere and for popcount.

To run doctests, run: ``PYTHONPATH=`pwd` sage -t probabilities.py``

"""
from mpmath import mp

import math
import numpy as np
from collections import namedtuple
from functools import partial
from memoize import memoize
from scipy.integrate import quad
from scipy.special import betainc

Probabilities = namedtuple(
    "Probabilities", ("d", "n", "k", "gr", "ngr", "pf", "ngr_pf", "gr_pf", "rho", "eta", "beta", "prec")
)


def _quad(f, a, b, epsrel=1e-10):
    """
    Integrate ``f`` over ``[a, b]`` with QUADPACK, using a breakpoint at the peak of ``f``.

    A single ``mp.quad`` call does not converge for these sharply peaked integrands.  Raises
    ``ArithmeticError`` if QUADPACK's error estimate exceeds the requested relative error.
    """
    a, b = float(a), float(b)
    if b <= a:
        return mp.mpf(0)
    xs = np.linspace(a, b, 65)[1:-1]
    peak = xs[np.argmax([float(f(x)) for x in xs])]
    r, err = quad(f, a, b, points=[peak], epsabs=0, epsrel=epsrel, limit=200)
    if not err <= epsrel * abs(r):
        raise ArithmeticError("quad on [%g, %g] gave %g with error estimate %g" % (a, b, r, err))
    return mp.mpf(r)


def C(d, theta, integrate=False, prec=None):
    """
    The probability that some v from the sphere has angle at most θ with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians
    :param: compute via explicit integration
    :param: precision to use

    EXAMPLE::

        sage: C(80, pi/3)
        mpf('1.0042233739846629e-6')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            r = (
                1
                / mp.sqrt(mp.pi)
                * mp.gamma(d / 2)
                / mp.gamma((d - 1) / 2)
                * mp.quad(lambda x: mp.sin(x) ** (d - 2), (0, theta), error=True)[0]
            )
        else:
            r = mp.betainc((d - 1) / 2, 1 / 2.0, x2=mp.sin(theta) ** 2, regularized=True) / 2
        return r


def A(d, theta, prec=53):
    """
    The density of the event that some v from the sphere has angle θ with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians

    :param: compute via explicit integration
    :param: precision to use

    EXAMPLES::

        sage: A(80, pi/3)
        mpf('4.7395659506025816e-5')

        sage: A(80, pi/3) * 2*pi/100000
        mpf('2.9779571143234787e-9')

        sage: C(80, pi/3+pi/100000) - C(80, pi/3-pi/100000)
        mpf('2.9779580567976835e-9')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        r = 1 / mp.sqrt(mp.pi) * mp.gamma(d / 2) / mp.gamma((d - 1) / 2) * mp.sin(theta) ** (d - 2)
        return r


@memoize
def log2_sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return (d / 2 * mp.log(mp.pi, 2) + 1) / mp.gamma(d / 2)


@memoize
def sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return 2 ** (d / 2 * mp.log(mp.pi, 2) + 1) / mp.gamma(d / 2)


@memoize
def W(d, alpha, beta, theta, integrate=True, prec=None):
    assert alpha <= mp.pi / 2
    assert beta <= mp.pi / 2
    assert 0 >= (mp.cos(beta) - mp.cos(alpha) * mp.cos(theta)) * (mp.cos(beta) * mp.cos(theta) - mp.cos(alpha))

    if theta >= alpha + beta:
        return mp.mpf(0.0)

    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        alpha = mp.mpf(alpha)
        beta = mp.mpf(beta)
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            c = mp.atan(mp.cos(alpha) / (mp.cos(beta) * mp.sin(theta)) - 1 / mp.tan(theta))

            e = float(d) - 2

            def f(x, t):
                y = max(-1.0, min(1.0, t / math.tan(x)))
                return math.sin(x) ** e * betainc(e / 2, 0.5, math.sin(math.acos(y)) ** 2)

            ta, tc = float(mp.tan(theta - c)), float(mp.tan(c))
            S_alpha = _quad(lambda x: f(x, ta), theta - c, alpha) / 2
            S_beta = _quad(lambda x: f(x, tc), c, beta) / 2

            return (S_alpha + S_beta) * sphere(d - 1) / sphere(d)
        else:
            # Wedge volume formula from Lemma 2.2 of [BDGL16] Anja Becker, Léo Ducas, Nicolas Gama,
            # Thijs Laarhoven. "New directions in nearest neighbor searching with applications to
            # lattice sieving." SODA 2016. https://eprint.iacr.org/2015/1128
            # g_sq = (mp.cos(alpha)**2 + mp.cos(beta)**2 -
            # 2*mp.cos(alpha)*mp.cos(beta)*mp.cos(theta))/mp.sin(theta)**2
            # log2_A = mp.log(g_sq, 2) - 2*mp.log(1-g_sq, 2)
            # r = (d-4) * mp.log(mp.sqrt(1-g_sq), 2) + log2_A - 2*mp.log(d-4, 2) + log2_sphere(d-2) - log2_sphere(d)
            # return 2**r
            raise NotImplementedError("Results don't match.")


@memoize
def binomial(n, i):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return mp.binomial(n, i)


@memoize
def P(n, k, theta, prec=None):
    """
    Probability that two vectors with angle θ pass a popcount filter

    :param n: number of popcount vectors
    :param k: number of popcount tests required to pass
    :param theta: angle in radians

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        # binomial cdf for 0 <= successes <= k
        # r = 0
        # for i in range(k):
        #     r += binomial(n, i) * (theta/mp.pi)**i * (1-theta/mp.pi)**(n-i)
        # r = mp.betainc(n-k, k+1, x2=1-(theta/mp.pi), regularized=True)
        # NOTE: This routine uses obscene precision

        def _betainc(a, b, x2):
            return (
                x2 ** a
                * mp.hyp2f1(
                    a, 1 - b, a + 1, x2, maxprec=2 ** mp.ceil(2 * mp.log(n, 2)), maxterms=2 ** mp.ceil(mp.log(n, 2))
                )
                / a
                / mp.beta(a, b)
            )

        r = _betainc(n - k, k + 1, x2=1 - (theta / mp.pi))
        return r


def pf(d, n, k, beta=None, lb=None, ub=None, beta_and=False, prec=None):
    """
    Let `Pr[P_{k,n}]` be the probability that a popcount filter passes.  We assume the probability
    is over the vectors `u,v`. Let `¬G` be the event that two random vectors are not Gauss reduced.

    We start with Pr[P_{k,n}]::

        sage: pf(80, 128, 40)
        mpf('0.00031063713572376122')

        sage: pf(80, 128, 128)
        mpf('1.0000000000000002')

    Pr[P_{k,n} ∧ ¬G]::

        sage: pf(80, 128, 40, ub=mp.pi/3)
        mpf('3.3598092589552732e-7')

    Pr[¬G]::

        sage: pf(80, 128, 128, ub=mp.pi/3)
        mpf('1.0042233739846644e-6')

        sage: ngr_pf(80, 128, 128)
        mpf('1.0042233739846644e-6')

        sage: ngr(80)
        mpf('1.0042233739846629e-6')

    Pr[Pr_{k,n} ∧ G]::

        sage: pf(80, 128, 40, lb=mp.pi/3)
        mpf('0.00031030115479786595')

    Pr[G]::

        sage: pf(80, 128, 128, lb=mp.pi/3)
        mpf('0.99999899577662632')

        sage: gr_pf(80, 128, 128)
        mpf('0.99999899577662632')

        sage: gr(80)
        mpf('0.99999899577662599')

    Pr[P_{k,n} | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3)
        mpf('0.019786655048072234')

    Pr[P_{k,n}  ∧ ¬G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, ub=mp.pi/3)
        mpf('0.00077177364924089652')

    Pr[¬G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, ub=mp.pi/3)
        mpf('0.0021964683579090904')
        sage: ngr_pf(80, 128, 128, beta=mp.pi/3)
        mpf('0.0021964683579090904')
        sage: ngr(80, beta=mp.pi/3)
        mpf('0.0021964683579090904')

    Pr[Pr_{k,n} ∧ G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, lb=mp.pi/3)
        mpf('0.019014953591444488')

        sage: gr_pf(80, 128, 40, beta=mp.pi/3)
        mpf('0.019014953591444488')

    Pr[G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, lb=mp.pi/3)
        mpf('0.99780353164285229')
        sage: gr_pf(80, 128, 128, beta=mp.pi/3)
        mpf('0.99780353164285229')
        sage: gr(80, beta=mp.pi/3)
        mpf('0.9978035316420909')

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param lb: lower bound of integration (see above)
    :param ub: upper bound of integration (see above)
    :param beta_and: return Pr[P_{k,n} ∧ C(w,β)] instead of Pr[P_{k,n} | C(w,β)]
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if lb is None:
            lb = 0
        if ub is None:
            ub = mp.pi
        if beta is None:
            return _quad(lambda x: P(n, k, x) * A(d, x), lb, ub)
        else:
            num = _quad(lambda x: P(n, k, x) * W(d, beta, beta, x) * A(d, x), lb, min(ub, 2 * beta))
            if not beta_and:
                den = _quad(lambda x: W(d, beta, beta, x) * A(d, x), 0, 2 * beta)
            else:
                den = 1
            return num / den


ngr_pf = partial(pf, lb=0, ub=mp.pi / 3)
gr_pf = partial(pf, lb=mp.pi / 3)


def ngr(d, beta=None, prec=None):
    """
    Probability that two random vectors (in a cap parameterised by β) are not Gauss reduced.

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if beta is None:
            return C(d, mp.pi / 3)
        elif beta < mp.pi / 6:
            return mp.mpf(1.0)
        else:
            # Pr[¬G ∧ E]
            num = _quad(lambda x: W(d, beta, beta, x) * A(d, x), 0, mp.pi / 3)
            # Pr[E]
            den = _quad(lambda x: W(d, beta, beta, x) * A(d, x), 0, 2 * beta)
            # Pr[¬G | E] = Pr[¬G ∧ E]/Pr[E]
            return num / den


def gr(d, beta=None, prec=None):
    """
    Probability that two random vectors (in a cap parameterised by β) are Gauss reduced.

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        return 1 - ngr(d, beta, prec)


def probabilities(d, n, k, beta=None, prec=None):
    """
    Useful probabilities.

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec

    with mp.workprec(prec):
        pf_ = pf(d, n, k, beta=beta, prec=prec)
        ngr_ = ngr(d, beta=beta, prec=prec)
        ngr_pf_ = ngr_pf(d, n, k, beta=beta, prec=prec)
        gr_pf_ = gr_pf(d, n, k, beta=beta, prec=prec)
        rho = 1 - ngr_pf_ / pf_
        eta = 1 - ngr_pf_ / ngr_

        probs = Probabilities(
            d=d,
            n=n,
            k=k,
            ngr=ngr_,
            gr=1 - ngr_,
            pf=pf_,
            gr_pf=gr_pf_,
            ngr_pf=ngr_pf_,
            rho=rho,
            eta=eta,
            beta=beta,
            prec=prec,
        )
        return probs
