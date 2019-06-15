# -*- coding: utf-8 -*-
from mpmath import mp
from probabilities_estimates import C


def list_sizef(d):
    # TODO: This doesn't use the ± trick
    return int(mp.ceil(1/C(d, mp.pi/3)))


def list_size_growthf(probs):
    # TODO: See e-mail from 15/06/2019
    return int(mp.ceil(mp.sqrt(1/(1-probs.eta))))


cnkf = list_size_growthf
