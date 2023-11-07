{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0> y;
  array[N] int<lower=0> n;
{% endblock %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ binomial_logit(n, mu_star_j);
{% endblock likelihood %}
{% endblock %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector<lower=0, upper=1>[N] mu_j = inv_logit(mu_star_j);
  vector<lower=0, upper=1>[2] mu = inv_logit(mu_star);
  real mu_diff = mu[1] - mu[2];
  array[N] int<lower=0> y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(nn in 1:N)
    y_rep[nn] = binomial_rng(n[nn], mu_j[nn]);
{% endblock %}
{% endblock %}
  
