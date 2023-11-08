import numpy as np
import pandas as pd
import arviz as az
import os
import glob

from abayes import ABayes
from abayes._globals import CACHE_LOCATION

SEED = 1234
rng = np.random.default_rng(SEED)

N = 1000
mu = [0, 1]
sigma = [0.2, 1]
y1 = rng.normal(size=N, loc=mu[0], scale=sigma[0]) 
y2 = rng.normal(size=N, loc=mu[1], scale=sigma[1]) 

cmdstan_kwargs = {"iter_warmup": 250, "iter_sampling": 250}

ab = ABayes(force_compile=True, seed=SEED, prior_only=True)

def test_abayes_fit_prior_predictive():
    assert ab.fit(data=(y1, y2), **cmdstan_kwargs)
    draws = ab.draws()
    assert np.isclose(draws["mu_diff"].mean(), 0, atol=1e-1)
    assert np.isclose(draws["sigma_diff"].mean(), 0, atol=1e-1)

