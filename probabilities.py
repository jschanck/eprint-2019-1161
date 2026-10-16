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
from scipy.special import betainc, betaln, gammaln

Probabilities = namedtuple(
    "Probabilities", ("d", "n", "k", "gr", "ngr", "pf", "ngr_pf", "gr_pf", "rho", "eta", "beta", "prec")
)


# Relative error of the integrals.  The inner integrals in ``W`` are more accurate than the outer
# integrals that use them.
EPSREL = 1e-10
EPSREL_INNER = 1e-12


def _log_quad(logf, a, b, epsrel=EPSREL, peak=None):
    """
    Return log ∫_a^b exp(logf(x)) dx with relative error ``epsrel`` (``-inf`` if ``b <= a``).

    The integrands are far below the smallest double (e.g. ``W(1024, β, β, θ) ≈ 1e-188``), so we divide
    by their value at the peak and integrate in double precision with QUADPACK, using the peak as a
    breakpoint.  The peak is found on a grid unless given.  If ``logf(peak)`` is ``-inf`` for a given peak,
    the integral is zero.

    :raises ArithmeticError: if the integrand vanishes on the grid or QUADPACK's error estimate exceeds
        ``epsrel`` times the result.

    """
    a, b = float(a), float(b)
    if b <= a:
        return -math.inf
    if peak is None:
        peak = max((a + (b - a) * i / 64 for i in range(1, 64)), key=logf)
        m = logf(peak)
        if m == -math.inf:
            raise ArithmeticError("integrand vanishes on [%g, %g]" % (a, b))
    else:
        m = logf(peak)
        if m == -math.inf:
            return -math.inf
    points = [peak] if a < peak < b else None
    r, err = quad(lambda x: math.exp(logf(x) - m), a, b, points=points, epsabs=0, epsrel=epsrel, limit=200)
    if not err <= epsrel * r:
        raise ArithmeticError("integral over [%g, %g] did not converge: %g ± %g" % (a, b, r, err))
    return m + math.log(r)


def _log_betainc(a, b, z):
    """
    Return log I_z(a, b), the regularized incomplete beta function, also where it underflows.
    """
    if z <= 0:
        return -math.inf
    if z >= 1:
        return 0.0
    r = betainc(a, b, z)
    if r > 1e-200:
        return math.log(r)
    # I_z(a, b) = z^a⋅(1-z)^b / (a⋅B(a, b)) ⋅ 2F1(a+b, 1; a+1; z), see DLMF 8.17.8.  For such small I_z,
    # z < a/(a+b) and the terms of the hypergeometric series decrease geometrically.
    s = t = 1.0
    j = 0
    while t > 1e-17 * s:
        t *= (a + b + j) * z / (a + 1 + j)
        s += t
        j += 1
        if j > 10 ** 6:
            raise ArithmeticError("I_%g(%g, %g) did not converge" % (z, a, b))
    return a * math.log(z) + b * math.log1p(-z) - math.log(a) - betaln(a, b) + math.log(s)


def _log_sphere_ratio(d):
    # log(sphere(d-1)/sphere(d)) = log(Γ(d/2) / (√π⋅Γ((d-1)/2)))
    return gammaln(d / 2) - gammaln((d - 1) / 2) - math.log(math.pi) / 2


def _log_A(d, theta):
    s = math.sin(theta)
    if s <= 0:
        return -math.inf
    return _log_sphere_ratio(d) + (d - 2) * math.log(s)


def _log_P(n, k, theta):
    if k >= n:
        return 0.0
    # binomial cdf for 0 <= successes <= k, i.e. I_{1-θ/π}(n-k, k+1)
    return _log_betainc(n - k, k + 1, 1 - theta / math.pi)


@memoize
def _log_W(d, alpha, beta, theta):
    if theta >= alpha + beta:
        return -math.inf

    # c = atan(cos(α)/(cos(β)⋅sin(θ)) - cot(θ)), rewritten to avoid cancellation for small θ
    num = math.cos(beta) * math.sin(theta / 2) ** 2 - math.sin((alpha + beta) / 2) * math.sin((alpha - beta) / 2)
    c = math.atan(2 * num / (math.cos(beta) * math.sin(theta)))

    def log_S(c0, h):
        # log ∫_{c0}^{c0+h} sin(x)^(d-2)⋅I_z((d-2)/2, 1/2) dx with z = 1 - tan(c0)^2/tan(x)^2.  We integrate over
        # u = x - c0 so that z is accurate when x is close to c0.  Where I_z is close to 1, z is so close to 1 that
        # 1 - z is inaccurate, so we use I_z(a, b) = 1 - I_{1-z}(b, a) and compute 1 - z directly.  The integrand
        # increases with x, so it peaks at u = h.
        def logf(u):
            x = c0 + u
            z = math.sin(u) * math.sin(2 * c0 + u) / (math.sin(x) * math.cos(c0)) ** 2
            if z <= 0:
                return -math.inf
            if betainc((d - 2) / 2, 0.5, z) <= 0.5:
                log_I = _log_betainc((d - 2) / 2, 0.5, z)
            else:
                log_I = math.log1p(-betainc(0.5, (d - 2) / 2, (math.tan(c0) / math.tan(x)) ** 2))
            return (d - 2) * math.log(math.sin(x)) + log_I

        return _log_quad(logf, 0, h, EPSREL_INNER, peak=h)

    log_S_alpha = log_S(theta - c, alpha - (theta - c))
    log_S_beta = log_S(c, beta - c)
    return float(np.logaddexp(log_S_alpha, log_S_beta)) - math.log(2) + _log_sphere_ratio(d)


@memoize
def _log_int(d, n, k, beta, lb, ub):
    """
    Return log ∫_lb^ub P(n, k, x)⋅W(d, β, β, x)⋅A(d, x) dx, leaving out ``P`` if ``n`` is ``None`` and ``W`` if
    ``beta`` is ``None``.  Arguments are floats, so that calls with the same arguments hit the cache.
    """

    def logf(x):
        r = _log_A(d, x)
        if n is not None:
            r += _log_P(n, k, x)
        if beta is not None:
            r += _log_W(d, beta, beta, x)
        return r

    if beta is not None:
        ub = min(ub, 2 * beta)
    return _log_quad(logf, lb, ub)


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
            r = mp.exp(_log_quad(lambda x: _log_A(float(d), x), 0, theta))
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


def W(d, alpha, beta, theta, integrate=True, prec=None):
    """
    The probability that a random w from the sphere has angle at most α with u and at most β with v,
    where u and v have angle θ.  The integral is computed in double precision with relative error about
    ``EPSREL_INNER``.
    """
    assert alpha <= mp.pi / 2
    assert beta <= mp.pi / 2
    assert 0 >= (mp.cos(beta) - mp.cos(alpha) * mp.cos(theta)) * (mp.cos(beta) * mp.cos(theta) - mp.cos(alpha))

    if theta >= alpha + beta:
        return mp.mpf(0.0)

    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if integrate:
            return mp.exp(_log_W(d, float(alpha), float(beta), float(theta)))
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


def P(n, k, theta, prec=None):
    """
    Probability that two vectors with angle θ pass a popcount filter

    :param n: number of popcount vectors
    :param k: number of popcount tests required to pass
    :param theta: angle in radians

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        return mp.exp(_log_P(n, k, float(theta)))


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
        mpf('0.0021964683571495717')

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
        mpf('0.99780353164285041')

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param lb: lower bound of integration (see above)
    :param ub: upper bound of integration (see above)
    :param beta_and: return Pr[P_{k,n} ∧ C(w,β)] instead of Pr[P_{k,n} | C(w,β)]
    :param prec: precision of the result; the integrals are computed in double precision with relative
        error about ``EPSREL``

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        lb = 0.0 if lb is None else float(lb)
        ub = math.pi if ub is None else float(ub)
        if beta is None:
            return mp.exp(_log_int(d, n, k, None, lb, ub))
        else:
            beta = float(beta)
            num = _log_int(d, n, k, beta, lb, ub)
            if beta_and:
                return mp.exp(num)
            den = _log_int(d, None, None, beta, 0.0, math.pi)
            return mp.exp(num - den)


ngr_pf = partial(pf, lb=0, ub=mp.pi / 3)
gr_pf = partial(pf, lb=mp.pi / 3)


def ngr(d, beta=None, prec=None):
    """
    Probability that two random vectors (in a cap parameterised by β) are not Gauss reduced.

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: precision of the result; the integrals are computed in double precision with relative
        error about ``EPSREL``

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if beta is None:
            return C(d, mp.pi / 3)
        elif beta < mp.pi / 6:
            return mp.mpf(1.0)
        else:
            beta = float(beta)
            # Pr[¬G ∧ E]
            num = _log_int(d, None, None, beta, 0.0, math.pi / 3)
            # Pr[E]
            den = _log_int(d, None, None, beta, 0.0, math.pi)
            # Pr[¬G | E] = Pr[¬G ∧ E]/Pr[E]
            return mp.exp(num - den)


def gr(d, beta=None, prec=None):
    """
    Probability that two random vectors (in a cap parameterised by β) are Gauss reduced.

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: precision of the result; the integrals are computed in double precision with relative
        error about ``EPSREL``

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
    :param prec: precision of the result; the integrals are computed in double precision with relative
        error about ``EPSREL``

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
