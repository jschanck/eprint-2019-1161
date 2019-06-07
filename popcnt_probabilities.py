# -*- coding: utf-8 -*-

from mpmath import mp
from functools import wraps


def memoize(function):
    memo = {}

    @wraps(function)
    def wrapper(*args):
        try:
            return memo[args]
        except KeyError:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


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
            r = (1/mp.sqrt(mp.pi) *
                 mp.gamma(d/2) / mp.gamma((d-1)/2) *
                 mp.quad(lambda x: mp.sin(x)**(d-2), (0, theta), error=True)[0])
        else:
            r = mp.betainc((d-1)/2, 1/2., x2=mp.sin(theta)**2, regularized=True)/2
        return r


def A(d, theta, prec=53):
    """
    The probability that some v from the sphere has angle θ with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians
    :param: compute via explicit integration
    :param: precision to use

    EXAMPLE::

        sage: A(80, pi/3)
        mpf('4.7395659506025816e-5')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        r = (1/mp.sqrt(mp.pi) *
             mp.gamma(d/2) / mp.gamma((d-1)/2) *
             mp.sin(theta)**(d-2))
        return r


@memoize
def sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return 2**(d/2*mp.log(mp.pi, 2) + 1) / mp.gamma(d/2)


def W(d, alpha, beta, theta, prec=None):
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        alpha = mp.mpf(alpha)
        beta  = mp.mpf(beta)
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        c = mp.atan(mp.cos(alpha)/(mp.cos(beta)*mp.sin(theta)) - 1/mp.tan(theta))

        def f_alpha(x):
            return mp.sin(x)**(d-2) * mp.betainc((d-2)/2, 1/2.,
                                                 x2=mp.sin(mp.acos(mp.tan(theta-c)/mp.tan(x)))**2,
                                                 regularized=True)

        def f_beta(x):
            return mp.sin(x)**(d-2) * mp.betainc((d-2)/2, 1/2.,
                                                 x2=mp.sin(mp.acos(mp.tan(c)/mp.tan(x)))**2,
                                                 regularized=True)

        S_alpha = mp.quad(f_alpha, (theta-c, alpha), error=True)[0]/2
        S_beta  = mp.quad(f_beta,  (c,       beta),  error=True)[0]/2

        return (S_alpha + S_beta) * sphere(d-1) / sphere(d)


@memoize
def binomial(n, i):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return mp.binomial(n, i)


# NOTE: hardcoding 53 here
def P(n, k, theta, nmk=True, prec=53):
    """
    Probability that two vectors with angle θ pass a popcount filter

    :param n: number of popcount vectors
    :param k: number of popcount tests required to pass
    :param theta: angle in radians
    :param nmk: consider also hamming weight n-k

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        r = 0
        for i in range(k):
            r += binomial(n, i) * (theta/mp.pi)**i * (1-theta/mp.pi)**(n-i)
        if nmk:
            r *= 2
        return r


def popcount_pass(d, n, k, beta=None, prec=None):
    with mp.workprec(prec):
        if beta is None:
            return mp.quad(lambda x: P(n, k, x)*A(d, x), (0, mp.pi), error=True)[0]
        else:
            num = mp.quad(lambda x: P(n, k, x)*W(d, beta, beta, x)*A(d, x), (0, 2*beta), error=True)[0]
            den = mp.quad(lambda x:            W(d, beta, beta, x)*A(d, x), (0, 2*beta), error=True)[0]
            return num/den


def not_gauss_reduced(d, beta=None, prec=None):
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if beta is None:
            return 2*C(d, mp.pi/3)  # NOTE 2*C_d here
        else:
            # Pr[-G ∧ E]
            num = 2*mp.quad(lambda x: W(d, beta, beta, x)*A(d, x), (0, mp.pi/3), error=True)[0]
            # Pr[E]
            den = mp.quad(lambda x:   W(d, beta, beta, x)*A(d, x), (0, 2*beta), error=True)[0]
            # Pr[-G | E] = Pr[-G ∧ E]/Pr[E]
            return num/den
