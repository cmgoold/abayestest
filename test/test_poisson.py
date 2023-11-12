import numpy as np
import pandas as pd
import arviz as az
import os
import glob

from abayestest import ABayesTest
from abayestest._globals import CACHE_LOCATION

SEED = 1234
rng = np.random.default_rng(SEED)

N = 1000
mu = [2, 4]
y1 = rng.poisson(size=N, lam=mu[0])
y2 = rng.poisson(size=N, lam=mu[1])

cmdstan_kwargs = {"iter_warmup": 250, "iter_sampling": 250}

ab = ABayesTest(likelihood="poisson", force_compile=True, seed=SEED)


def test_abayestest_poisson_fit():
    ab.fit(data=(y1, y2))
    draws = ab.draws()
    assert np.isclose(mu[0] - mu[1], draws["mu_diff"].mean(), rtol=1e-1)
