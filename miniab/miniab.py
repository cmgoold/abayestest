from __future__ import annotations

from jinja2 import Environment, PackageLoader
from jinja2.exceptions import TemplateNotFound
import os
from pathlib import Path
import cmdstanpy as csp

from .templates.distributions import LIKELIHOODS

ROOT = Path(__file__).parent.parent.resolve()
CACHE_LOCATION =  ROOT / ".miniab"

if not os.path.exists(CACHE_LOCATION):
    os.mkdir(CACHE_LOCATION)

DEFAULT_PRIORS = {"mu": "normal(0, 1)", "sigma": "normal(0, 1)"}

ENVIRONMENT = Environment(loader=PackageLoader("miniab"))


class MiniAb(object):
    """The main A/B testing class.

    This class initializes a MiniAb object instance, given a specified
    likelihood function and a set of priors.

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, likelihood: str = "normal", priors: Priors = DEFAULT_PRIORS) -> None:
        self._likelihood = likelihood
        self._priors = priors
        self.model : CmdStanModel = self.compile()

    likelihood = property(lambda self: self._likelihood)
    priors = property(lambda self: self._priors)

    def fit(data: Union[Dict[str, DataTypes], pd.DataFrame], **cmdstanpy_kwargs) -> MiniAb:
        fit = self.model.sample(data=data, **kwargs)
        return self

    def compile(self) -> CmdStanModel:
        if self._hash() in os.listdir(CACHE_LOCATION):
            return csp.CmdStanModel(exe_file=str(CACHE_LOCATION) + "/" + self._hash())
        else:
            stan_file = str(CACHE_LOCATION) + "/" + self._hash() + ".stan"
            with open(stan_file, "w") as f:
                f.write(self._render_model())
            return csp.CmdStanModel(stan_file=stan_file)

    def _render_model(self) -> str:
        try:
            template = ENVIRONMENT.get_template("distributions/" + self._likelihood.lower() + ".stan")
        except TemplateNotFound:
            raise ValueError(f"Cannot build model for likelihood {self._likelihood}.\n"
                             f"Likelihoods available are {LIKELIHOODS}.")

        rendered = template.render(priors=self.priors)
        return rendered
        
    
    @property
    def samples(self) -> np.ndarray:
        return self.fit.stan_variables()

    @property
    def summary(self) -> PosteriorSummary:
        pass

    def _hash(self):
        return str(1234)

