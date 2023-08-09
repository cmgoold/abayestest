from __future__ import annotations

from jinja2 import Environment, PackageLoader

DEFAULT_PRIORS = {"mu": "normal(0, 1)", "sigma": "normal(0, 1)"}

LIKELIHOODS = [dist.replace(".stan", "") for dist in os.listdir("templates/distributions") if dist != "base.stan"]

ENVIRONMENT = Environment(loader=PackageLoader("miniab"))

class AB(object):
    """The main A/B testing class.

    This class initializes an AB object instance, given a specified
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

    def fit(data: Union[Dict[str, DataTypes], pd.DataFrame], **cmdstanpy_kwargs) -> AB:
        fit = self.model.sample(data=data, **kwargs)
        return self

    def compile(self) -> CmdStanModel:
        if self._hash() in os.listdir(CACHE_LOCATION):
            return csp.CmdStanModel(exe_file=self._hash())
        else:
            stan_file = open(CACHE_LOCATION / self._hash(), "w").write(self._render_model())
            return csp.CmdStanModel(stan_file=stan_file)

    def _render_model(self) -> str:
        try:
            template = env.get_template(self.likelihood.lower())
        except TemplateNotFoundError:
            raise ValueError(f"Cannot build model for likelihood {self.likelihood}.\n"
                             f"Likelihoods available are {LIKELIHOODS}.")

        rendered = template.render(priors=self.priors)
        return rendered
        
    
    @property
    def samples(self) -> np.ndarray:
        return self.fit.stan_variables()

    @property
    def summary(self) -> PosteriorSummary:
        pass

