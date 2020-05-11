#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from utils import bulk_create_and_store_bundles, bulk_cost_estimate
from cost import all_pairs, random_buckets, list_decoding, sieve_size, Metrics, SizeMetrics
from texconstsf import main as texconstsff
import click


@click.command()
@click.argument("d_min", type=int)
@click.argument("d_max", type=int)
@click.argument("d_stepsize", type=int)
@click.option("--jobs", default=4, help="number of jobs to run in parallel")
@click.option("--probabilities", default=False, help="compute and store probabilities", type=bool)
def runall(d_min=64, d_max=1024, d_stepsize=16, jobs=4, probabilities=False):
    if probabilities:
        D = range(d_min, 256 + 1, d_stepsize)
        N = [2 ** i - 1 for i in range(5, 16)]
        _ = bulk_create_and_store_bundles(D, N, BETA=[], ncores=jobs)

        D = range(256 + d_stepsize, d_max + 1, d_stepsize)
        N = [2 ** i - 1 for i in range(5, 14)]
        _ = bulk_create_and_store_bundles(D, N, BETA=[], ncores=jobs)

    bulk_cost_estimate(
        (all_pairs, random_buckets, list_decoding),
        range(d_min, d_max + 1, d_stepsize),
        metric=Metrics,
        ncores=jobs,
    )

    bulk_cost_estimate((sieve_size,), range(d_min, d_max + 1, 2), metric=SizeMetrics, ncores=jobs)

    texconstsff()


if __name__ == "__main__":
    runall()
