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

ab = ABayes(force_compile=True, seed=SEED)

def test_abayes_instance():
    assert isinstance(ab, ABayes)

def test_abayes_fit_from_tuple():
    assert ab.fit(data=(y1, y2), **cmdstan_kwargs)
    draws = ab.draws()
    assert np.isclose(draws["mu_diff"].mean(), mu[0] - mu[1], rtol=1e-1)
    assert np.isclose(draws["sigma_diff"].mean(), sigma[0] - sigma[1], rtol=1e-1)

def test_abayes_fit_from_dict():
    assert ab.fit(data={"y1": y1, "y2": y2}, **cmdstan_kwargs)
    draws = ab.draws()
    assert np.isclose(draws["mu_diff"].mean(), mu[0] - mu[1], rtol=1e-1)
    assert np.isclose(draws["sigma_diff"].mean(), sigma[0] - sigma[1], rtol=1e-1)

def test_abayes_methods():
    ab.fit(data=(y1, y2), **cmdstan_kwargs)
    assert isinstance(ab.draws(), dict)
    assert isinstance(ab.summary(), pd.DataFrame)
    assert isinstance(ab.inference_data, az.InferenceData)

def test_abayes_hash():
    ab2 = ABayes()
    assert ab2._hash() == ab._hash()
