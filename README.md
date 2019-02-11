Files for quantum-sieving-revisited.tex
=======================================

The .qasm files in this directory can be compiled into either png of pdf files by running ./qasm2png file.qasm or ./qasm2pdf file.qasm respectively.
However qasm2pdf seems quite shell specific and qasm2png requires some non standard commands, so the pdfs required to compile the main document have been added.


popcnt.py
========

This python file has all the necessary functions for experimentally vindicating the probabilities arrived at in the main document. A general overview, for a dimension ``d`` lattice, using ``n`` vectors to create SimHashes and a threshold ``k``, and using ``db`` many vectors to get experimental results, is given by

.. code-block:: bash

    ipython popcnt.py d db n k

where the output is read as ``probability:    experimental: experimental_value    estimate: estimated_value``, with key ``(n)gr = (not) Gauss reduced``, ``(n)pf = (did not) pass the filter``, underscores mean ``and`` (e.g. ``gr_pf = Gauss reduced and passed the filter``) and ``/`` mean conditional probabilities (e.g. ``gr_pf/pf = Gauss reduced given passed the filter``).

One can also do quick tests using ``filter_wrapper`` and ``gauss_wrapper`` as follows

.. code-block:: python

    >>> from popcnt import filter_wrapper, gauss_wrapper
    >>> filter_wrapper(range(64, 129, 8), 5000, 256, 96)    # same order as __main__ above, but this time can take lists except for db
    >>> gauss_wrapper(range(64, 129, 8), 5000)              # just takes d (potentially as a list) and db


optimise_popcnt.py
=================

This python file creates and saves estimated probabilities for all possible popcnt parameters (up to caller defined limit of vectors used to create the SimHashes) for a given dimension. It also allows optimisation over a (caller defined) function of the probabilities estimated, for a given dimension.

To generate all probabilities in dimension ``d`` using up to ``n_max`` vectors to create the SimHashes and save them as a pickle in ./probabilities do

.. code-block:: python

    >>> from optimise_popcnt import create_estimates
    >>> create_estimates(d, max_popcnt_num=n_max)

To optimise over a function (not the naive default), where a higher value is better, do

.. code-block:: python

    >>> from optimise_popcnt import maximise_optimising_function as max_opt
    >>> def f(gr, pf, gr_pf, gr_npf, ngr_pf, ngr_npf):
            # return some function of the arguments
    >>> max_opt(d, f=f, max_popcnt_num=n_max)                               # where create_estimates(d, max_popcnt_num=n_max) has been called

and the maximum of the function f over the input popcnt parameters will be returned and the tuple ``(d, n, k)`` which first achieved them.
