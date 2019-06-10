# -*- coding: utf-8 -*-
from mpmath import mp
from popcnt_estimates import Probabilities


def split_interval(int_l, int_u, slices):
    """
    Splits closed interval [intl_l, int_u] into ``slices`` number of slices of
    equal size

    :param int_l: the lower bound of the interval
    :param int_u: the upper bound of the interval
    :param slices: the number of slices of equal size to split interval into

    :returns: list of the values that define the slices

    """
    int_l = mp.mpf(int_l)
    int_u = mp.mpf(int_u)
    intervals = [int_l]
    for i in range(1, slices + 1):
        intervals += [int_l + mp.mpf(i)*(int_u - int_l)/mp.mpf(slices)]
    return intervals


def probability_gadget(dp1, n, k, int_l=0, int_u=mp.pi, use_filt=True,
                       pass_filt=True):
    """
    A function for computing the various probabilities we are interested in.

    If we are not considering the filter at all (e.g. when calculating Gauss reduced probabilities)
    set ``use_filt`` to ``False``, in which case ``pass_filt``, ``n`` and ``k`` are ignored.

    If we want to calculate probabilities when pairs do not pass the filter, set ``pass_filt`` to
    ``False``.

    :param dp1: the dimension of real space, i.e. `d+1`
    :param n: the number of vectors with which to make the SimHash
    :param k: the acceptance/rejection threshold for popcnts
    :param int_l: the lower bound of the integration in [0, π]
    :param int_u: the upper bound of the integration in [0, π]
    :param use_filt: boolean whether to consider the filter
    :param pass_filt: boolean whether to consider passing/failing the filter

    :returns: the chosen probability

    """
    dp1 = mp.mpf(dp1)
    d = dp1 - 1
    n = mp.mpf(n)
    k = mp.mpf(k)

    # if the integrations are not accurate, increase the intervals
    interval = split_interval(int_l, int_u, 1)

    if use_filt:
        if pass_filt:
            coeffs  = [mp.binomial(n, i) for i in range(0, k)]
            coeffs += [mp.mpf('0')] * int(n - (2 * k) + 1)
            coeffs += [mp.binomial(n, i) for i in range(n - k + 1, n + 1)]
        else:
            coeffs  = [mp.mpf('0')] * int(k)
            coeffs += [mp.binomial(n, i) for i in range(k, n - k + 1)]
            coeffs += [mp.mpf('0')] * int(k)

        prob = 0
        for i in range(n + 1):
            co = coeffs[i]
            i = mp.mpf(i)
            if co == 0:
                continue

            def f(x):
                return co * mp.sin(x)**(d-1) * ((x/mp.pi)**i) * ((1-(x/mp.pi))**(n-i)) # noqa
            prob += mp.quad(f, interval, maxdegree=50000, error=True)[0]

    else:

        def f(x):
            return mp.sin(x)**(d-1)
        prob = mp.quad(f, interval, maxdegree=50000, error=True)[0]

    def normaliser(dp1):
        """
        The normalisation constant (dependent on dp1) has a closed form!

        .. note:: we are interested in the relative surface area of
        (hyper)spheres on the surface of `S^{d}`, hence `d - 1`.

        :param dp1: the dimension of real space
        :returns: the normalisation constant for the integral estimates
        """
        dp1 = mp.mpf(dp1) - mp.mpf('2')
        norm = mp.mpf('1')
        if dp1 % 2 == 0:
            for i in range(1, dp1 + 1):
                if i % 2 == 0:
                    norm *= mp.mpf(i)**mp.mpf('-1')
                else:
                    norm *= mp.mpf(i)
            norm *= mp.pi
        else:
            for i in range(2, dp1 + 1):
                if i % 2 == 0:
                    norm *= mp.mpf(i)
                else:
                    norm *= mp.mpf(i)**mp.mpf('-1')
            norm *= mp.mpf(2)
        return mp.mpf(1)/norm

    return normaliser(dp1) * prob


def probabilities(d, n, k, prec=None):
    """
    For a given real dimension, number of popcnt test vectors and a k, determines all useful
    probabilities.

    :param d: the dimension of real space
    :param n: the number of vectors with which to make the SimHash
    :param k: the acceptance/rejection k for popcnts
    :param efficient: if ``True`` only compute probs needed for (3)

    :returns: an OrderedDict with keys the names of a given probability
    """
    prec = prec if prec else mp.prec

    with mp.workprec(prec):
        gr = probability_gadget(d, n, k, int_l=mp.pi/3, int_u=(2*mp.pi)/3, use_filt=False)
        pf = probability_gadget(d, n, k)
        ngr_pf = mp.mpf('2')*probability_gadget(d, n, k, int_u=mp.pi/3)
        gr_pf = probability_gadget(d, n, k, int_l=mp.pi/3, int_u=(2*mp.pi)/3)

        probs = Probabilities(d=d, n=n, k=k,
                              gr=gr,
                              pf=pf,
                              gr_pf=gr_pf,
                              ngr_pf=ngr_pf,
                              beta=None, prec=prec)

        return probs
