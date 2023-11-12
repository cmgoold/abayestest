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
Priors = Dict[str, str]


class ABayesTest(object):
    """The main A/B testing class.

    This class initializes an ABayesTest object instance, given a specified
    likelihood function and a set of priors. If desired, a prior
    predictive simulation can be run by the `prior_only` argument
    to the constructor.

    Attributes
    ------------

    likelihood : str
        The chosen likelihood function.

    priors : str
        The prior structure. Defaults to DEFAULT PRIORS, above.

    prior_only: bool
        Should a prior predictive simulation be run?

    cmdstan_mcmc : cmdstanpy.CmdStanMCMC
        The fitted cmdstanpy.CmdStanMCMC object.

    num_draws : int
        A helper property to get the number of posterior
        or prior-predictive draws sampled.

    seed : int
        The chosen seed.
    """

    def __init__(
        self,
        likelihood: Literal[LIKELIHOODS] = "normal",
        priors: Optional[Priors] = None,
        prior_only: bool = False,
        seed: int = None,
        force_compile: bool = False,
    ) -> None:
        """

        Parameters
        ----------

        likelihood : str
            The likelihood function. One of LIKELIHOODS.

        priors : Priors
            The dictionary of priors for the `mu_star` and/or
            `sigma_star` parameters, depending on the likelihood.
            Can be `None`, which defaults to the DEFAULT_PRIORS
            for the likelihood.

        prior_only : bool
            Should a prior predictive simulation be run? Defaults
            to `False`.

        seed : int
            The random seed. Can be `None`.

        force_compile : bool
            Should compilation be forced? In some cases,
            we might not want to abayestest to used
            cached models, in which case the model is
            re-compiled. Defaults to `False`.
        """
        self._likelihood = likelihood.lower()
        if self._likelihood not in LIKELIHOODS:
            raise ValueError(
                f"Unknown likelihood {self.likelihood}. Available likelihoods are {LIKELIHOODS}."
            )
        self._priors = (
            {**DEFAULT_PRIORS[self._likelihood], **priors}
            if priors is not None
            else DEFAULT_PRIORS[self._likelihood]
        )
        self._prior_only = prior_only
        self.model: csp.CmdStanModel = self.compile(force=force_compile)
        self._fit: csp.CmdStanMCMC = None
        self._seed = seed

    likelihood = property(lambda self: self._likelihood)
    priors = property(lambda self: self._priors)
    prior_only = property(lambda self: self._prior_only)
    cmdstan_mcmc = property(lambda self: self._fit)
    num_draws = property(lambda self: self._fit.num_draws_sampling * self._fit.chains)
    seed = property(lambda self: self._seed)

    def fit(self, data: DataTypes, **cmdstanpy_kwargs) -> abayestest:
        """Fit the model to the data.

        The data passed to this function should be in the form
        of a two-key dictionary or a tuple of length 2
        for each group's data, e.g. (y1, y2).
        The exception is the binomial likelihood, where
        the additional data `n` for the binomial PMF's
        size variable needs to also be supplied. In this
        case, y1 and y2 above are assumed to be a tuple
        of (n, y).

        Parameters
        ----------
        data : DataTypes
            The data, either as a dictionary or tuple.

        cmdstanpy_kwargs : Any
            Arbitrary keyword arguments to cmdstanpy.CmdStanModel.sample.

        Returns
        -------
        self

        Raises
        -----
        A ValueError is raised if the data is not an iterable.
        """

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
        self._fit = self.model.sample(
            data=clean_data,
            **{"seed": self.seed, "show_console": True, **cmdstanpy_kwargs},
        )
        return self

    def compile(self, force: bool = False) -> CmdStanModel:
        """Compile the model.

        Parameters
        -----------
        force : bool
            Should compilation be forced? If `False`,
            previously run models are pulled
            from the cache.

        Returns
        -------
        An instance of cmdstanpy.CmdStanModel
        """

        stan_file = self._hash() + ".stan"
        if force or stan_file not in os.listdir(CACHE_LOCATION):
            stan_file_path = str(CACHE_LOCATION) + "/" + stan_file
            with open(stan_file_path, "w") as f:
                f.write(self._render_model())
            return csp.CmdStanModel(stan_file=stan_file_path)
        else:
            return csp.CmdStanModel(exe_file=str(CACHE_LOCATION) + "/" + self._hash())

    def _render_model(self) -> str:
        """Render the model from the jinja2 template."""
        try:
            template = ENVIRONMENT.get_template(
                "distributions/" + self._likelihood.lower() + ".stan"
            )
        except TemplateNotFound:
            raise ValueError(
                f"Cannot build model for likelihood {self._likelihood}.\n"
                f"Likelihoods available are {LIKELIHOODS}."
            )

        rendered = template.render(priors=self.priors, sample=int(not self.prior_only))
        return rendered

    @property
    def inference_data(self):
        """Access the Arviz.InferenceData object."""
        self._check_fit_exists()
        return az.from_cmdstanpy(self.cmdstan_mcmc)

    @lru_cache
    def draws(self) -> np.ndarray:
        """Access the posterior draws."""
        self._check_fit_exists()
        return self._fit.stan_variables()

    @lru_cache
    def summary(self) -> pd.DataFrame:
        """Summarize the MCMC results.

        This function returns an Arviz summary
        object for the key parameters in the
        model.
        """
        self._check_fit_exists()
        variables = ["mu", "mu_diff", "mu_star", "mu_star_diff"]
        if self._likelihood == "normal":
            variables += ["sigma", "sigma_diff", "sigma_star", "sigma_star_diff"]
        return az.summary(self.inference_data, var_names=variables)

    def diagnose(self) -> str:
        """Diagnose the fit.

        This uses the diagnostic function from cmdstanpy/cmdstan directly.
        """
        self._check_fit_exists()
        print(self._fit.diagnose())

    def compare_conditions(self) -> str:
        """Compare the A and B conditions.

        This function outputs a simple summary string
        comparing the distributions of the fitted parameters.
        """
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
            ""
            if not report_sigma
            else f"\n{(sum(sigma_a_minus_b < 0)/self.num_draws) * 100:.2f}% of the posterior differences for sigma favour condition B.",
        )

    def _hash(self):
        """Hash the model, along with the priors, prior_only statement and likelihood."""
        return md5(
            json.dumps(tuple((self.priors, self.prior_only, self.likelihood))).encode(
                "utf-8"
            )
        ).hexdigest()

    def _check_fit_exists(self) -> Union[None, Exception]:
        if self._fit is None:
            raise AttributeError("The model has not been fit yet.")
        else:
            return True
