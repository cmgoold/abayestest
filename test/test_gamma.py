import numpy as np
import pandas as pd
import arviz as az
import os
import glob

from ABayes import ABayes
from ABayes._globals import CACHE_LOCATION

SEED = 1234
rng = np.random.default_rng(SEED)

N = 1000
mu = [2, 5]
sigma = [0.2, 0.5]
y1 = rng.gamma(size=N, scale=sigma[0]**2 / mu[0], shape=mu[0]**2 / sigma[0]**2)
y2 = rng.gamma(size=N, scale=sigma[1]**2 / mu[1], shape=mu[1]**2 / sigma[1]**2)

cmdstan_kwargs = {"iter_warmup": 250, "iter_sampling": 250}

ab = ABayes(likelihood="gamma", force_compile=True, seed=SEED)

def test_ABayes_gamma_fit():
    ab.fit(data=(y1, y2))
    draws = ab.draws()
    assert np.isclose(mu[0] - mu[1], draws["mu_diff"].mean(), rtol=1e-1) 
    assert np.isclose(sigma[0] - sigma[1], draws["sigma_diff"].mean(), rtol=1e-1) 


