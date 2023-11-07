import numpy as np
import pandas as pd
import arviz as az
import os
import glob

from abayes import ABayes
from abayes._globals import CACHE_LOCATION

SEED = 1234
rng = np.random.default_rng(SEED)

def logit(x):
    return np.log(x) - np.log(1 - x)

N = 1000
mu = [0.6, 0.9]
n = rng.choice(range(70, 100), N)
y1 = rng.binomial(n=n, size=N, p=mu[0])
y2 = rng.binomial(n=n, size=N, p=mu[1])

cmdstan_kwargs = {"iter_warmup": 250, "iter_sampling": 250}

ab = ABayes(likelihood="binomial", force_compile=True, seed=SEED)

def test_abayes_bernoulli_fit():
    ab.fit(data=((n, y1), (n, y2)))
    draws = ab.draws()
    assert np.isclose(mu[0] - mu[1], draws["mu_diff"].mean(), rtol=1e-1) 


