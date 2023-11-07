{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0, upper=1> y;
{% endblock %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ bernoulli_logit(mu_star_j);
{% endblock likelihood %}
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
  
