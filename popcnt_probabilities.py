# -*- coding: utf-8 -*-

from mpmath import mp
from functools import wraps, partial


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
        r = (1/mp.sqrt(mp.pi) *
             mp.gamma(d/2) / mp.gamma((d-1)/2) *
             mp.sin(theta)**(d-2))
        return r


@memoize
def log2_sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return (d/2*mp.log(mp.pi, 2) + 1) / mp.gamma(d/2)


@memoize
def sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return 2**(d/2*mp.log(mp.pi, 2) + 1) / mp.gamma(d/2)


def W(d, alpha, beta, theta, integrate=True, prec=None):
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        alpha = mp.mpf(alpha)
        beta  = mp.mpf(beta)
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
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
        else:
            # Wedge volume formula from Lemma 2.2 of [BDGL16] Anja Becker, Léo Ducas, Nicolas Gama,
            # Thijs Laarhoven. "New directions in nearest neighbor searching with applications to
            # lattice sieving." SODA 2016. https://eprint.iacr.org/2015/1128
            # g_sq = (mp.cos(alpha)**2 + mp.cos(beta)**2 - 2*mp.cos(alpha)*mp.cos(beta)*mp.cos(theta))/mp.sin(theta)**2
            # log2_A = mp.log(g_sq, 2) - 2*mp.log(1-g_sq, 2)
            # r = (d-4) * mp.log(mp.sqrt(1-g_sq), 2) + log2_A - 2*mp.log(d-4, 2) + log2_sphere(d-2) - log2_sphere(d)
            # return 2**r
            raise NotImplementedError("Results don't match.")


@memoize
def binomial(n, i):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return mp.binomial(n, i)


# NOTE: hardcoding 53 here
def P(n, k, theta, nmk=False, prec=53):
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


def pf(d, n, k, beta=None, lb=None, ub=None, beta_and=False, prec=None):
    """
    Let `Pr[P_{k,n}]` be the probability that a popcount filter passes.  We assume the probability
    is over the vectors `u,v`. Let `¬G` be the event that two random vectors are not Gauss reduced.

    We start with Pr[P_{k,n}]::

        sage: pf(80, 128, 40)
        mpf('0.00031063713572376074')

        sage: pf(80, 128, 128)
        mpf('0.99999999999999989')

    Pr[P_{k,n} ∧ ¬G]::

        sage: pf(80, 128, 40, ub=mp.pi/3)
        mpf('3.3598092589552753e-7')

    Pr[¬G]::

        sage: pf(80, 128, 128, ub=mp.pi/3)
        mpf('1.0042233739846655e-6')

        sage: pf_ngr(80, 128, 128)
        mpf('1.0042233739846655e-6')

        sage: ngr(80)
        mpf('1.0042233739846629e-6')

    Pr[Pr_{k,n} ∧ G]::

        sage: pf(80, 128, 40, lb=mp.pi/3)
        mpf('0.0003103011547978659')

    Pr[G]::

        sage: pf(80, 128, 128, lb=mp.pi/3)
        mpf('0.99999899577662754')

        sage: pf_gr(80, 128, 128)
        mpf('0.99999899577662754')

        sage: gr(80)
        mpf('0.99999899577662599')

    Pr[P_{k,n} | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3)  ## TODO: mpc!
        mpc(real='0.019786655048072255', imag='0.0')

    Pr[P_{k,n}  ∧ ¬G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, ub=mp.pi/3) ## TODO: mpc!
        mpc(real='0.00077177364924089728', imag='0.0')

    Pr[¬G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, ub=mp.pi/3) ## TODO: mpc!
        mpc(real='0.0021964683579090943', imag='0.0')
        sage: pf_ngr(80, 128, 128, beta=mp.pi/3)  ## TODO: mpc!
        mpc(real='0.0021964683579090943', imag='0.0')
        sage: ngr(80, beta=mp.pi/3)  ## TODO: mpc!
        mpc(real='0.0021964683579090904', imag='0.0')

    Pr[Pr_{k,n} ∧ G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, lb=mp.pi/3)
        mpc(real='0.019014953591444523', imag='0.0')

        sage: pf_gr(80, 128, 40, beta=mp.pi/3)
        mpc(real='0.019014953591444523', imag='0.0')

    Pr[G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, lb=mp.pi/3) ## TODO: mpc!
        mpc(real='0.99780353164285163', imag='0.0')
        sage: pf_gr(80, 128, 128, beta=mp.pi/3) ## TODO: mpc!
        mpc(real='0.99780353164285163', imag='0.0')
        sage: gr(80, beta=mp.pi/3) ## TODO: mpc!
        mpc(real='0.9978035316420909', imag='0.0')

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
            return mp.quad(lambda x: P(n, k, x)*A(d, x), (lb, ub), error=True)[0]
        else:
            num = mp.quad(lambda x: P(n, k, x)*W(d, beta, beta, x)*A(d, x), (lb, min(ub, 2*beta)), error=True)[0]
            if not beta_and:
                den = mp.quad(lambda x:            W(d, beta, beta, x)*A(d, x), (0, 2*beta), error=True)[0]
            else:
                den = 1
            return num/den


pf_ngr = partial(pf, lb=0, ub=mp.pi/3)
pf_gr  = partial(pf, lb=mp.pi/3)


def ngr(d, beta=None, prec=None):
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if beta is None:
            return C(d, mp.pi/3)
        else:
            # Pr[¬G ∧ E]
            num = mp.quad(lambda x: W(d, beta, beta, x)*A(d, x), (0, mp.pi/3), error=True)[0]
            # Pr[E]
            den = mp.quad(lambda x: W(d, beta, beta, x)*A(d, x), (0, 2*beta), error=True)[0]
            # Pr[¬G | E] = Pr[¬G ∧ E]/Pr[E]
            return num/den


def gr(d, beta=None, prec=None):
    return 1-ngr(d, beta, prec)
