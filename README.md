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
\mu_{A} &= 0, \quad \sigma_{A} = 0.2 \\
\mu_{B} &= 1, \quad \sigma_{B} = 1
\end{align}
$$

That is, both groups' data are normally distributed
with locations, `0` and `1`, and scales
`0.2` and `1`, respectively.
Thus, there is a true difference of means of `-1` and
a true difference of scales of `-0.8`. Here's the Python
code:

```python
import numpy as np

from abayes import ABayes

SEED = 1234
rng = np.random.default_rng(SEED)

N = 100
mu = [0, 1]
sigma = [0.2, 1]
y_a = rng.normal(size=N, loc=mu[0], scale=sigma[0]) 
y_b = rng.normal(size=N, loc=mu[1], scale=sigma[1]) 
```

We then initialize an `ABayes` object with the default options
(normal likelihood, default priors) and fit the model, passing
the data in as a tuple:

```python
ab = ABayes(seed=SEED)
ab.fit(data=(y_a, y_b))
```

The model will run in Stan and return `self`.
You can access the `cmdstanpy.CmdStanMCMC` object
itself using `ab.cmdstan_mcmc`. 
For instance, we can use `cmdstanpy`'s diagnostic
function to check for any convergence problems:

```python
ab.diagnose()
```

which returns:

```
Checking sampler transitions treedepth.
Treedepth satisfactory for all transitions.

Checking sampler transitions for divergences.
No divergent transitions found.

Checking E-BFMI - sampler transitions HMC potential energy.
E-BFMI satisfactory.

Effective sample size satisfactory.

Split R-hat values satisfactory all parameters.

Processing complete, no problems detected.
```

indicating not problems.

To take a quick
look at the results, run `ab.summary()`, which returns
a summary Pandas `DataFrame` straight from [`Arviz`](
https://github.com/arviz-devs/arviz
):

```
                  mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu[0]            0.026  0.023  -0.018    0.067      0.000    0.000    4655.0    3215.0    1.0
mu[1]            1.059  0.105   0.851    1.249      0.001    0.001    5554.0    3166.0    1.0
mu_diff         -1.033  0.108  -1.222   -0.820      0.001    0.001    5566.0    3225.0    1.0
mu_star[0]       0.026  0.023  -0.018    0.067      0.000    0.000    4655.0    3215.0    1.0
mu_star[1]       1.059  0.105   0.851    1.249      0.001    0.001    5554.0    3166.0    1.0
mu_star_diff    -1.033  0.108  -1.222   -0.820      0.001    0.001    5566.0    3225.0    1.0
sigma[0]         0.229  0.016   0.199    0.259      0.000    0.000    4938.0    3202.0    1.0
sigma[1]         1.046  0.077   0.904    1.190      0.001    0.001    4530.0    2968.0    1.0
sigma_diff      -0.817  0.078  -0.973   -0.681      0.001    0.001    4504.0    3051.0    1.0
sigma_star[0]   -1.478  0.071  -1.616   -1.349      0.001    0.001    4938.0    3202.0    1.0
sigma_star[1]    0.042  0.073  -0.101    0.174      0.001    0.001    4530.0    2968.0    1.0
sigma_star_diff -1.520  0.101  -1.709   -1.334      0.001    0.001    4755.0    3271.0    1.0
```

ABayes always uses the parameter `mu` to refer to 
the vector of group-specific locations, or other non-normal
distribution's canonincal parameters (e.g. the Poisson
rate parameter; see below). Dispersion parameters,
such as the normal distribution's scale parameter,
are referred to as `sigma`.

The parameters suffixed with `_star` are unconstrained
parameters, which ABayes uses for estimation under-the-hood.
More details about the parameter transformations and 
likelihood parameterizations are given below, but
for the normal distribution, `mu = mu_star` and `sigma_star = log(sigma)`.
Conditions A and B are always indexed as `0` and `1`
in the Python outputs.
The additional variables `mu_diff` and `sigma_diff` (and
the `_star` companions) give
the difference in posterior distributions between groups 1 and 2
(i.e. `mu[0] - mu[1]` using Python's zero-indexing).
As we can see, these recover the data-generating assumptions above,
with posterior means close to `-1` and `-0.8` for the means
and standard deviations, respectively.

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
from scipy.stats import gaussian_kde

def density(x):
    limits = x.min(), x.max()
    grid = np.linspace(*limits, 1000)
    return grid, gaussian_kde(x)(grid)

mu_diff = ab.draws()["mu_diff"]

plt.plot(*density(mu_diff), color="#0492c2", lw=4)
plt.axvline(0, ls=":", color="gray")
plt.xlabel("score")
plt.ylabel("density")
plt.title("posterior of condition A - B")
```

![](doc/a-minus-b.png)

The `ABayes` class also contains a handy method
to report the distribution of
differences in the posteriors
between conditions called
`compare_conditions`, which
tells us that:

```
100.00% of the posterior differences 
favour condition B.
```

## Posterior predictive distribution
ABayes automatically calculates the posterior predictive
distribution of the data, which is accessible in
the posterior draws object under the key `y_rep`. 
This array is in long form, where group A and B's
predictions are stacked on top of each other.
Using the example above, we can inspect this
distribution using a small bit of manipulation
of the draws:

```python
y_rep_raw = ab.draws()["y_rep"]
y_reps = y_rep_raw[:, :N], y_rep_raw[:, N:]
ys = y_a, y_b

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    a_or_b = (1 - i) * "A" + i * "B"
    grid, samples = density(y_reps[i].flatten())
    ax[i].plot(grid, samples, color="#0492c2", lw=3, label="Posterior predictive")
    ax[i].plot(ys[i], [0.01]*len(ys[i]), '|', color="black", label="raw")
    ax[i].set_title(a_or_b)
    ax[i].set_xlabel("score")
    ax[i].set_ylabel("density")
    if not i:
        ax[i].legend(frameon=False, loc="upper right")
```

![](doc/ppd.png)

# Likelihood functions

Currently, ABayes supports normal, lognormal, gamma,
Bernoulli, binomial, and Poisson distributions.

For non-normal likelihood functions, ABayes
calculates the differences in canonical
parameters on both unconstrained and 
original scales, which are detailed below:

| Distribution | Parameterization | Link function   | Unconstrained name | Original name |
| ------------ | ---------------- | --------------- | ------------------ | ------------- |
| normal       | mean/variance    | identity        | `mu_star`, `sigma_star`      | `mu`, `sigma`   |
| Bernoulli    | probability      | logit           | `mu_star`               | `mu`     |
| binomial     | logit            | `mu`               | `mu_prob`     |
| lognormal    | log              | `mu`               | `mu_pos`      |
| gamma        | log              | `mu`               | `mu`          |


## Under the hood 
We can in inspect the likelihood distribution and priors via 
an `ABayes` instance's properties:

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

