from __future__ import annotations

from typing import Optional, List, Union, Tuple, Dict
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader
from jinja2.exceptions import TemplateNotFound
from pathlib import Path
from functools import cached_property, lru_cache
from hashlib import md5
import json
import os

import arviz as az
import cmdstanpy as csp

from .templates.distributions import LIKELIHOODS
from ._globals import CACHE_LOCATION, ROOT

__all__ = [
    "ABayesTest",
    "DEFAULT_PRIORS",
]

DEFAULT_PRIORS = {
    "normal": {"mu_star": "normal(0, 1)", "sigma_star": "normal(0, 1)"},
    "lognormal": {"mu_star": "normal(0, 1)", "sigma_star": "normal(0, 1)"},
    "gamma": {"mu_star": "normal(0, 1)", "sigma_star": "normal(0, 1)"},
    "poisson": {"mu_star": "normal(0, 1)"},
    "bernoulli": {"mu_star": "normal(0, 1)"},
    "binomial": {"mu_star": "normal(0, 1)"},
}

ENVIRONMENT = Environment(loader=PackageLoader("abayestest"))

VectorTypes = Union[List, np.ndarray]
DataTypes = Union[Dict[str, VectorTypes], Tuple[VectorTypes, ...]]
Priors = Dict[str, Tuple[str, ...]]

class ABayesTest(object):
    """The main A/B testing class.

    This class initializes an ABayesTest object instance, given a specified
    likelihood function and a set of priors.
    """

    def __init__(self, likelihood: Literal[LIKELIHOODS] = "normal", priors: Optional[Priors] = None, prior_only: bool = False, seed: int = None, force_compile: Optiona[bool] = None) -> None:
        self._likelihood = likelihood.lower()
        if self._likelihood not in LIKELIHOODS:
            raise ValueError(f"Unknown likelihood {self.likelihood}. Available likelihoods are {LIKELIHOODS}.")
        self._priors = priors if priors is not None else DEFAULT_PRIORS[self._likelihood]
        self._prior_only = prior_only
        self.model : csp.CmdStanModel = self.compile(force=force_compile)
        self._fit: csp.CmdStanMCMC = None
        self._seed = seed

    likelihood = property(lambda self: self._likelihood)
    priors = property(lambda self: self._priors)
    prior_only = property(lambda self: self._prior_only)
    cmdstan_mcmc = property(lambda self: self._fit)
    num_draws = property(lambda self: self._fit.num_draws_sampling * self._fit.chains)
    seed = property(lambda self: self._seed)

    def fit(self, data: DataTypes, **cmdstanpy_kwargs) -> abayestest:
        if not hasattr(data, "__iter__"):
            raise ValueError("Data passed to abayestest.fit must be an iterable.")
        if isinstance(data, Dict):
            y1, y2 = data.values()
        else:
            y1, y2 = data
        if self.likelihood == "binomial":
            (n1, y1), (n2, y2) = y1, y2
        y = np.hstack([y1, y2])
        if self.likelihood == "binomial":
            n = np.hstack([n1, n2])
        _j = [1] * len(y1) + [2] * len(y2)
        clean_data = {"N": len(y1) + len(y2), "j": _j, "y": y}
        if self.likelihood == "binomial":
            clean_data["n"] = n
        self._fit = self.model.sample(data=clean_data, **{"seed": self.seed, "show_console": True,  **cmdstanpy_kwargs})
        return self

    def compile(self, force: bool = False) -> CmdStanModel:
        stan_file = self._hash() + ".stan"
        if force or stan_file not in os.listdir(CACHE_LOCATION):
            stan_file_path = str(CACHE_LOCATION) + "/" + stan_file
            with open(stan_file_path, "w") as f:
                f.write(self._render_model())
            return csp.CmdStanModel(stan_file=stan_file_path)
        else:
            return csp.CmdStanModel(exe_file=str(CACHE_LOCATION) + "/" + self._hash())

    def _render_model(self) -> str:
        try:
            template = ENVIRONMENT.get_template("distributions/" + self._likelihood.lower() + ".stan")
        except TemplateNotFound:
            raise ValueError(f"Cannot build model for likelihood {self._likelihood}.\n"
                             f"Likelihoods available are {LIKELIHOODS}.")

        rendered = template.render(priors=self.priors, sample=int(not self.prior_only))
        return rendered
        
    @property
    def inference_data(self):
        self._check_fit_exists()
        return az.from_cmdstanpy(self.cmdstan_mcmc)
    
    @lru_cache
    def draws(self) -> np.ndarray:
        self._check_fit_exists()
        return self._fit.stan_variables()

    @lru_cache
    def summary(self) -> pd.DataFrame:
        self._check_fit_exists()
        variables = ["mu", "mu_diff", "mu_star", "mu_star_diff"]
        if self._likelihood == "normal":
            variables += ["sigma", "sigma_diff", "sigma_star", "sigma_star_diff"]
        return az.summary(self.inference_data, var_names=variables)

    def diagnose(self) -> str:
        self._check_fit_exists()
        print(self._fit.diagnose())

    def compare_conditions(self) -> str:
        self._check_fit_exists()
        mu_a_minus_b = self.draws()["mu_diff"]
        if "sigma_diff" in self.draws():
            sigma_a_minus_b = self.draws()["sigma_diff"]
            report_sigma = 1
        else:
            sigma_a_minus_b = 0
            report_sigma = 0
        return print(
            f"{(sum(mu_a_minus_b < 0)/self.num_draws) * 100:.2f}% of the posterior differences for mu favour condition B.",
            "" if not report_sigma else f"\n{(sum(sigma_a_minus_b < 0)/self.num_draws) * 100:.2f}% of the posterior differences for sigma favour condition B."
        )
    
    def _hash(self):
        return md5(json.dumps(tuple((self.priors, self.prior_only, self.likelihood))).encode("utf-8")).hexdigest()

    def _check_fit_exists(self) -> Union[None, Exception]:
        if self._fit is None:
            raise AttributeError("The model has not been fit yet.")
        else:
            return True


