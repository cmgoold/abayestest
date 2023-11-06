![build](https://github.com/cmgoold/miniab/actions/workflows/test.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ABayes

ABayes is a lightweight Python package for performing Bayesian AB testing.
Computations are run using [Stan](
https://mc-stan.org
) via [`cmdstanpy`](
https://github.com/stan-dev/cmdstanpy
), and [`jinja2`](
https://github.com/pallets/jinja/
) is used
as a backend templating engine
to construct the Stan model files.

## Installation

To install, use:

```bash
python3.10 -m pip install git+ssh://git@github.com/cmgoold/abayes.git
```

Installing ABayes will also create a local cache folder for storing
Stan model objects, which is `.abayes` in the repository root.

### CmdStan
ABayes requires a working `cmdstan` installation. The easiest
way to download `cmdstan` is [via `cmdstanpy`](
https://mc-stan.org/cmdstanpy/installation.html#function-install-cmdstan
).

## Simple API

The simplest use-case is running a comparison
between two sets of approximately normally-distributed
data sets. First, let's sample some fake data, where
we have two groups with the following data generating
process:

$$
\begin{align}
y_{ij} &\sim \mathrm{Normal}(\mu_{j}, \sigma_{j})\\
\mu_{A} &= 0\\
\sigma_{A} &= 0.2 \\
\mu_{B} &= 1\\
\sigma_{B} &= 1
\end{align}
$$

That is, both groups' data are normally distributed
with locations, `0` and `1`, and scales
`0.2` and `1`, respectively.
Thus, there is a true difference of means of `1` and
a true difference of scales of `0.8`. Here's the Python
code:

```python
import numpy as np

from abayes import ABayes

SEED = 1234
rng = np.random.default_rng(SEED)

N = 50
mu = [0, 1]
sigma = [0.2, 1]
y_a = rng.normal(size=N, loc=mu[0], scale=sigma[0]) 
y_b = rng.normal(size=N, loc=mu[1], scale=sigma[1]) 
```

We then initialize an `ABayes` object with the default options
(normal likelihood, default priors) and fit the model, passing
the data in as a tuple:

```python
ab = ABayes()
ab.fit(data=(y_a, y_b), seed=SEED)
```

The model will run in Stan and return `self`.
You can access the `cmdstanpy.CmdStanMCMC` object
itself using `ab.cmdstan_mcmc`. To take a quick
look at the results, run `ab.summary`, which returns
a summary Pandas `DataFrame` straight from [`Arviz`](
https://github.com/arviz-devs/arviz
):

```
             mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu[0]      -0.026  0.034  -0.091    0.036      0.001    0.000    4408.0    2992.0    1.0
mu[1]       1.351  0.148   1.068    1.619      0.002    0.002    4401.0    3014.0    1.0
mu_diff    -1.377  0.152  -1.663   -1.086      0.002    0.002    4426.0    3121.0    1.0
sigma[0]    0.238  0.025   0.192    0.284      0.000    0.000    3814.0    2860.0    1.0
sigma[1]    1.055  0.108   0.869    1.265      0.002    0.001    3850.0    2518.0    1.0
sigma_diff -0.817  0.110  -1.013   -0.605      0.002    0.001    3791.0    2720.0    1.0
```

ABayes always uses the terms `mu` and `sigma` to refer to 
vectors of group-specific means and standard deviations,
and condition A and B are always indexed as `0` and `1`
in the outputs..
The additional variables `mu_diff` and `sigma_diff` give
the difference in posterior distributions between groups 1 and 2
(i.e. `mu[0] - mu[1]` using Python's zero-indexing).
As we can see, these recover the data-generating assumptions above.

Using the estimated quantities, users can calculate
any quantities or metrics that are meaningful
to the AB test being performed. For instance,
the probability that condition B scores greater than
A is the proportion of the posterior distribution
of `mu[1] - mu[0]` that is greater than zero,
which in this case is 100%, as can be inferred
from the `mu_diff` distribution directly:

```python
import matplotlib.pyplot as plt

mu_diff = ab.draws["mu_diff"]

plt.hist(mu_diff, bins=40, color="skyblue", alpha=0.5)
plt.axvline(0, ls=":")
plt.xlabel("score")
plt.ylabel("density")
plt.title("posterior of condition B - A")
```

![](docs/b-minus-a.png)


## Under the hood 
We can in inspect the likelihood distribution and priors via 
an `ABayes` instance's properties:
properties:

```python
ab.likelihood, ab.priors
Out[3]: ('normal', {'mu': 'normal(0, 1)', 'sigma': 'normal(0, 1)'})
```

The priors correspond to both groups (i.e. the Stan data is assumed in
long-format and the prior statements are vectorized). Currently,
different priors for each group is not a supported feature.
By default, standard normal priors are set on the model parameters
(standard half-normal for standard deviations),
which are accessed via `abayes.DEFAULT_PRIORS`.
The prior text are just strings passed directly to Stan, so
users can subsititute with any distribution and constants they wish.

The `ab.model` attribute returns the `cmdstanpy.CmdStanModel` attribute,
which is stored in the cache location. The 'private' attribute `_render_model`
can be used, if interested, to see the raw Stan code:

```python
ab._render_model()
```

which returns:

```stan
/* Stan file generated by Conor Goold, 2023. 
 * This program is covered by an GNU license.
*/ 

data {
  /* data declarations */
  int<lower=0> N;
  array[N] int<lower=1, upper=2> j;
  vector[N] y;
}

transformed data {
  /* data transformations */
}

parameters {
  
  /* raw model parameters */
  vector[2] mu;
  vector<lower=0>[2] sigma;

}

transformed parameters {
  
  /* parameter transformations */
  // nb: no change of variables adjustments are made
  // to these parameters
  vector[N] mu_j = mu[j];
  vector<lower=0>[N] sigma_j = sigma[j];

}

model {

  
  /* priors */
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);


  /* likelihood statement */
  y ~ normal(mu_j, sigma_j);

}

generated quantities {

  
  /* declarations */
  real mu_diff;
  real sigma_diff;
  vector[N] y_rep;


  
  /* computations */
  mu_diff = mu[1] - mu[2];
  sigma_diff = sigma[1] - sigma[2];
  for(n in 1:N)
    y_rep[n] = normal_rng(mu_j[n], sigma_j[n]);

}
```

