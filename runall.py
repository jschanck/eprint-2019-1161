# -*- coding: utf-8 -*-
import sys
from utils import bulk_create_and_store_bundles, bulk_cost_estimate
from cost import all_pairs, random_buckets, list_decoding, sieve_size, Metrics, SizeMetrics
from texconstsf import main

NCORES = int(sys.argv[1])

D = range(64, 256+1, 16)
N = [2**i-1 for i in range(5, 16)]
_ = bulk_create_and_store_bundles(D, N, BETA=[], ncores=NCORES)

D = range(272, 1024+1, 16)
N = [2**i-1 for i in range(5, 14)]
_ = bulk_create_and_store_bundles(D, N, BETA=[], ncores=NCORES)

D = range(64, 1024+1, 16)
SIEVES = [all_pairs, random_buckets, list_decoding]
bulk_cost_estimate(SIEVES, D, metric=Metrics, ncores=NCORES)

D = range(2, 1024+1)
SIEVES = [sieve_size,]
bulk_cost_estimate(SIEVES, D, metric=SizeMetrics, ncores=NCORES)

main()
