{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0, upper=1> y;
{% endblock %}

{% block transformed_parameters %}
{% block tpar_declarations %}
  {{ super() }}
{% endblock tpar_declarations %}
{% block likelihood %}
  for(nn in 1:N)
    lp[nn] = bernoulli_logit_lpmf(y[nn] | mu_star_j[nn]);
{% endblock likelihood %}
{% endblock transformed_parameters %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block log_density %}
  {{ super() }}
{% endblock log_density %}
{% endblock %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector<lower=0, upper=1>[N] mu_j = inv_logit(mu_star_j);
  vector<lower=0, upper=1>[2] mu = inv_logit(mu_star);
  real mu_diff = mu[1] - mu[2];
  array[N] int<lower=0, upper=1> y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(n in 1:N)
    y_rep[n] = bernoulli_logit_rng(mu_star_j[n]);
{% endblock %}
{% endblock %}
  
