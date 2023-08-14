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
mu_prob = [0.6, 0.9]
mu = [logit(i) for i in mu_prob]
y1 = rng.binomial(n=1, size=N, p=mu_prob[0])
y2 = rng.binomial(n=1, size=N, p=mu_prob[1])

cmdstan_kwargs = {"iter_warmup": 250, "iter_sampling": 250}

ab = ABayes(likelihood="bernoulli", force_compile=True, seed=SEED)

def test_abayes_bernoulli_fit():
    ab.fit(data=(y1, y2))
    draws = ab.draws
    assert np.isclose(mu_prob[0] - mu_prob[1], draws["mu_prob_diff"].mean(), rtol=1e-1) 


